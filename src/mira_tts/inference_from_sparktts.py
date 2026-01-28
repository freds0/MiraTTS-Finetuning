import os
import re
import gc
import inspect
import argparse
import sys
from typing import Any, List, Optional
import torch
import soundfile as sf
from vllm import LLM, SamplingParams
from ncodec.codec import TTSCodec

# --- Environment Setup for Stability ---
# Forces the 'spawn' method to avoid deadlocks/crashes when vLLM shuts down.
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

# Regex patterns for token parsing
_RE_CTX = re.compile(r"context_token_(\d+)")
_RE_SEM = re.compile(r"bicodec_semantic_(\d+)")

def clear_cache():
    """Forces garbage collection and empties CUDA cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def _make_llm(**kwargs):
    """Safely initializes LLM by filtering supported arguments."""
    sig = inspect.signature(LLM.__init__)
    allowed = set(sig.parameters.keys())
    safe = {k: v for k, v in kwargs.items() if k in allowed}
    return LLM(**safe)

def tok_str(prefix: str, ids: torch.Tensor) -> str:
    """Converts tensor IDs to token string format."""
    ids = ids.detach().cpu().reshape(-1).tolist()
    return "".join(f"<|{prefix}_{int(i)}|>" for i in ids)

class MiraTTS:
    def __init__(
        self,
        model_dir: str,
        tp: int = 1,
        enable_prefix_caching: bool = True,
        dtype: str = "bfloat16",
        max_model_len: int = 2048,
        max_num_seqs: int = 4,
        gpu_memory_utilization: float = 0.6, # Set to 0.6 to leave VRAM for the Codec
        enforce_eager: bool = True,
    ):
        llm_kwargs = dict(
            model=model_dir,
            tensor_parallel_size=int(tp),
            dtype=dtype,
            enable_prefix_caching=bool(enable_prefix_caching),
            max_model_len=int(max_model_len),
            max_num_seqs=int(max_num_seqs),
            gpu_memory_utilization=float(gpu_memory_utilization),
            enforce_eager=bool(enforce_eager),
            trust_remote_code=True,
            disable_log_stats=True,
        )
        
        print(f"[MiraTTS] Initializing vLLM with GPU memory utilization: {gpu_memory_utilization*100}%")
        self.llm = _make_llm(**llm_kwargs)

        self.set_params()
        # Initialize the audio codec
        self.codec = TTSCodec()

    def set_params(
        self,
        top_p: float = 0.95,
        top_k: int = 50,
        temperature: float = 0.8,
        max_new_tokens: int = 1024,
        repetition_penalty: float = 1.2,
        min_p: float = 0.05,
    ):
        self.sampling_params = SamplingParams(
            top_p=float(top_p),
            top_k=int(top_k),
            temperature=float(temperature),
            max_tokens=int(max_new_tokens),
            repetition_penalty=float(repetition_penalty),
            min_p=float(min_p),
        )

    def encode_audio(self, audio_file: str):
        """Encodes the reference audio for voice cloning."""
        if not os.path.exists(audio_file):
            raise FileNotFoundError(f"Reference audio file not found: {audio_file}")
        return self.codec.encode(audio_file)

    def _vllm_generate_texts(self, prompts: List[str]) -> List[str]:
        """Internal method to call vLLM generation."""
        try:
            outs = self.llm.generate(prompts, self.sampling_params)
        except TypeError:
            outs = self.llm.generate(prompts, sampling_params=self.sampling_params)

        texts = []
        for o in outs:
            texts.append(o.outputs[0].text if o.outputs else "")
        return texts

    def generate(self, text: str, context_tokens: Any):
        """Generates audio from text using the reference context."""
        
        formatted_prompt = self.codec.format_prompt(text, context_tokens, None)
        context_tokens_ids = (torch.tensor([int(token) for token in re.findall(r"context_token_(\d+)", context_tokens)]))
        context_tokens_str = tok_str("bicodec_global", context_tokens_ids)
        
        inputs = [
            "<|task_tts|>",
            "<|start_content|>",
            text,
            "<|end_content|>",
            "<|start_global_token|>",
            context_tokens_str,
            "<|end_global_token|>",
        ]
        
        inputs_str = "".join(inputs)
        generated_text = self._vllm_generate_texts([inputs_str])[0]

        pred_semantic_ids = (torch.tensor([int(token) for token in re.findall(r"bicodec_semantic_(\d+)", generated_text)]))
        
        if pred_semantic_ids.numel() == 0:
            print(f"[WARNING] No audio tokens generated for text: {text[:30]}...")
            return None

        generated_text_ids = tok_str("speech_token", pred_semantic_ids)
        context_tokens_orig = tok_str("context_token", context_tokens_ids)

        audio = self.codec.decode(generated_text_ids, context_tokens_orig)
        return audio

    def close(self):
        """Explicitly release resources."""
        if hasattr(self, 'llm'):
            del self.llm
        clear_cache()

def main():
    parser = argparse.ArgumentParser(description="MiraTTS TXT Batch Processor")
    
    # Required Arguments
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the LLM model directory")
    parser.add_argument("--txt", type=str, required=True, help="Path to input .txt file (one sentence per line)")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory where .wav files will be saved")
    parser.add_argument("--ref_audio", type=str, required=True, help="Path to reference audio file (for voice cloning)")
    
    # Optional Arguments
    parser.add_argument("--gpu_mem", type=float, default=0.6, help="GPU memory utilization (0.0 to 1.0). Default: 0.6")

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    tts = None
    try:
        # 1. Initialize Model
        print(f"[INFO] Loading model from: {args.checkpoint}")
        tts = MiraTTS(
            model_dir=args.checkpoint,
            gpu_memory_utilization=args.gpu_mem,
            max_model_len=2048
        )

        # 2. Process Reference Audio
        print(f"[INFO] Encoding reference audio: {args.ref_audio}")
        ctx = tts.encode_audio(args.ref_audio)

        # 3. Read TXT File
        prompts = []
        if not os.path.exists(args.txt):
            raise FileNotFoundError(f"Input text file not found: {args.txt}")

        print(f"[INFO] Reading texts from file: '{args.txt}'")
        with open(args.txt, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                clean_line = line.strip()
                # Only add if the line is not empty
                if clean_line:
                    prompts.append((i, clean_line))

        print(f"[INFO] Total sentences to process: {len(prompts)}")

        # 4. Generation Loop
        for original_idx, text in prompts:
            try:
                print(f"[{original_idx+1}/{len(lines)}] Generating: {text[:50]}...")
                
                audio = tts.generate(text, ctx)
                
                if audio is not None:
                    audio = audio.float()
                    # Filename format: 00000.wav (based on line number in txt)
                    filename = f"{original_idx:05d}.wav"
                    save_path = os.path.join(args.output_dir, filename)
                    
                    sf.write(save_path, audio.cpu().numpy(), 48000)
                
                # Periodic cache clearing
                if original_idx % 10 == 0:
                    clear_cache()

            except Exception as e:
                print(f"[ERROR] Failed to generate sentence index {original_idx}: {e}")
                continue

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    except Exception as e:
        print(f"\n[CRITICAL ERROR] {e}")
    finally:
        # 5. Clean Shutdown
        print("\n[INFO] Cleaning up resources...")
        if tts:
            tts.close()
        
        if torch.distributed.is_initialized():
            print("[INFO] Destroying distributed process group...")
            torch.distributed.destroy_process_group()
        
        print("[INFO] Done.")

if __name__ == "__main__":
    main()

