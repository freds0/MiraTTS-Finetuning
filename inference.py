#!/usr/bin/env python3
"""
Batch inference script for MiraTTS

Generates audio files from a list of sentences using a trained checkpoint.
Based on the inference code from MiraTTS_training.ipynb
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import librosa
import soundfile as sf
from huggingface_hub import snapshot_download

# Setup environment before importing unsloth
os.environ['UNSLOTH_FORCE_FLOAT32'] = '1'
os.environ['ACCELERATE_MIXED_PRECISION'] = 'no'

from unsloth import FastModel
from ncodec.codec import TTSCodec
from ncodec.encoder.model import audio_volume_normalize


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Batch inference for MiraTTS - Generate audio from text sentences",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with local checkpoint
  python inference.py \\
      --checkpoint ./checkpoints/ \\
      --sentences sentences.txt \\
      --reference reference.wav \\
      --output-dir ./generated_audio/

  # With HuggingFace model
  python inference.py \\
      --checkpoint YatharthS/MiraTTS \\
      --sentences sentences.txt \\
      --reference reference.wav \\
      --output-dir ./generated_audio/ \\
      --hf-token your_token_here

  # With custom generation parameters
  python inference.py \\
      --checkpoint ./checkpoints/ \\
      --sentences sentences.txt \\
      --reference reference.wav \\
      --output-dir ./generated_audio/ \\
      --temperature 0.9 \\
      --top-k 60

Sentences file format (one sentence per line):
  Hello, this is the first sentence.
  This is the second sentence.
  And this is the third one.
        """
    )

    # Required arguments
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint directory or HuggingFace model ID"
    )
    parser.add_argument(
        "--sentences",
        type=str,
        required=True,
        help="Path to text file with sentences (one per line)"
    )
    parser.add_argument(
        "--reference",
        type=str,
        required=True,
        help="Path to reference audio file for voice cloning"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save generated audio files"
    )

    # Optional authentication
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="HuggingFace token for accessing private repositories"
    )

    # Generation parameters
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature (default: 0.8)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k sampling (default: 50)"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Top-p (nucleus) sampling (default: 1.0)"
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.2,
        help="Repetition penalty (default: 1.2)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Maximum new audio tokens (default: 1024)"
    )

    # Output options
    parser.add_argument(
        "--prefix",
        type=str,
        default="audio",
        help="Prefix for output filenames (default: 'audio')"
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Starting index for output filenames (default: 0)"
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=48000,
        help="Output sample rate (default: 48000)"
    )

    return parser.parse_args()


def load_sentences(file_path: str) -> list:
    """Load sentences from a text file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        sentences = [line.strip() for line in f if line.strip()]
    return sentences


def setup_custom_encode(tts_codec):
    """Setup custom encode function for the codec (from notebook)"""
    @torch.inference_mode()
    def encode(audio, encode_semantic=True, duration=30.0):
        """Encodes audio file into speech tokens and context tokens"""
        self = tts_codec.audio_encoder

        audio = audio_volume_normalize(audio)
        ref_clip = self.get_ref_clip(audio)
        wav_ref = torch.from_numpy(ref_clip).unsqueeze(0).float()

        mel = self.mel_transformer(wav_ref).squeeze(1)
        new_arr = np.array(mel.transpose(1, 2).cpu())

        global_tokens = self.s_encoder.run(["global_tokens"], {"mel_spectrogram": new_arr})
        context_tokens = "".join([f"<|context_token_{i}|>" for i in global_tokens[0].squeeze()])

        if encode_semantic:
            feat = self.extract_wav2vec2_features(audio)
            speech_tokens = self.q_encoder.run(["semantic_tokens"], {"features": feat.cpu().detach().numpy()})
            speech_tokens = "".join([f"<|speech_token_{i}|>" for i in speech_tokens[0][0]])
            return speech_tokens, context_tokens
        else:
            return context_tokens

    tts_codec.audio_encoder.encode = encode
    tts_codec.audio_encoder.feature_extractor.config.output_hidden_states = True
    return tts_codec


def load_model(checkpoint_path: str, hf_token: str = None, max_seq_length: int = 1500):
    """Load model from local path or HuggingFace"""

    # Check if it's a local path
    if os.path.exists(checkpoint_path):
        print(f"Loading model from local path: {checkpoint_path}")
        model_path = checkpoint_path
    else:
        print(f"Downloading model from HuggingFace: {checkpoint_path}...")
        model_path = snapshot_download(checkpoint_path, token=hf_token)

    print("Loading model...")
    model, tokenizer = FastModel.from_pretrained(
        model_name=model_path,
        max_seq_length=max_seq_length,
        dtype=torch.float32,
        full_finetuning=True,
        load_in_4bit=False,
        torch_dtype='float32',
    )

    print("Model loaded successfully!")
    return model, tokenizer


def infer(model, tokenizer, tts_codec, text: str, audio_file: str,
          top_k=50, top_p=1.0, temperature=0.8, repetition_penalty=1.2,
          max_new_audio_tokens=1024, device='cuda:0'):
    """Generate speech from text using reference audio (from notebook)"""

    print(f"  [DEBUG] Text: {text[:50]}...")
    print(f"  [DEBUG] Audio file: {audio_file}")

    # Load reference audio
    audio_array, sr = librosa.load(audio_file, sr=16000)
    print(f"  [DEBUG] Reference audio loaded: shape={audio_array.shape}, dtype={audio_array.dtype}, sr={sr}")

    # Encode reference audio (get context tokens only)
    context_tokens = tts_codec.encode(audio_array)
    print(f"  [DEBUG] Context tokens: type={type(context_tokens)}, len={len(context_tokens) if isinstance(context_tokens, str) else 'N/A'}")
    print(f"  [DEBUG] Context tokens preview: {context_tokens[:100] if isinstance(context_tokens, str) else context_tokens}...")

    # Format prompt
    formatted_prompt = tts_codec.format_prompt(text, context_tokens, None)
    print(f"  [DEBUG] Formatted prompt: type={type(formatted_prompt)}, len={len(formatted_prompt)}")
    print(f"  [DEBUG] Prompt preview: {formatted_prompt[:200]}...")

    # Tokenize
    model_inputs = tokenizer([formatted_prompt], return_tensors="pt").to(device)
    print(f"  [DEBUG] Model inputs: input_ids shape={model_inputs.input_ids.shape}, dtype={model_inputs.input_ids.dtype}")

    # Generate
    print("  Generating token sequence...")
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_audio_tokens,
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )
    print("  Token sequence generated.")
    print(f"  [DEBUG] Generated IDs: shape={generated_ids.shape}, dtype={generated_ids.dtype}")

    # Decode tokens
    generated_ids_trimmed = generated_ids[:, model_inputs.input_ids.shape[1]:]
    print(f"  [DEBUG] Generated IDs trimmed: shape={generated_ids_trimmed.shape}")

    predicts_text = tokenizer.batch_decode(generated_ids_trimmed, skip_special_tokens=False)[0]
    print(f"  [DEBUG] Predicted text: type={type(predicts_text)}, len={len(predicts_text)}")
    print(f"  [DEBUG] Predicted text preview: {predicts_text[:200]}...")

    # Decode audio
    print("  Decoding audio...")
    print(f"  [DEBUG] Calling tts_codec.decode(predicts_text, context_tokens)")
    print(f"  [DEBUG]   predicts_text type={type(predicts_text)}, len={len(predicts_text)}")
    print(f"  [DEBUG]   context_tokens type={type(context_tokens)}, len={len(context_tokens) if isinstance(context_tokens, str) else 'N/A'}")

    audio = tts_codec.decode(predicts_text, context_tokens)
    print(f"  [DEBUG] Decoded audio: type={type(audio)}")
    if hasattr(audio, 'shape'):
        print(f"  [DEBUG] Audio shape={audio.shape}, dtype={audio.dtype}")
    if hasattr(audio, '__len__'):
        print(f"  [DEBUG] Audio len={len(audio)}")
    if hasattr(audio, 'min') and hasattr(audio, 'max'):
        print(f"  [DEBUG] Audio min={audio.min()}, max={audio.max()}")

    # Convert torch.Tensor to numpy float32
    if isinstance(audio, torch.Tensor):
        print(f"  [DEBUG] Converting torch.Tensor to numpy float32...")
        audio = audio.cpu().float().numpy()  # .float() converts to float32
        print(f"  [DEBUG] Converted: shape={audio.shape}, dtype={audio.dtype}")
    # Convert numpy float16 to float32 if needed
    elif hasattr(audio, 'dtype') and audio.dtype == np.float16:
        print(f"  [DEBUG] Converting numpy float16 to float32...")
        audio = audio.astype(np.float32)
        print(f"  [DEBUG] Converted: dtype={audio.dtype}")

    print(f"  [DEBUG] Final audio: type={type(audio)}, shape={audio.shape}, dtype={audio.dtype}")
    print(f"  [DEBUG] Final audio min={audio.min()}, max={audio.max()}")
    return audio


def save_audio(audio, output_path: str, sample_rate: int = 48000):
    """Save audio to file"""
    # Convert to float32 if needed (soundfile doesn't support float16)
    if hasattr(audio, 'dtype') and audio.dtype == np.float16:
        audio = audio.astype(np.float32)

    sf.write(output_path, audio, sample_rate)


def main():
    """Main batch inference function"""
    args = parse_args()

    print("=" * 70)
    print("MiraTTS Batch Inference")
    print("=" * 70)

    # Validate inputs
    if not os.path.exists(args.sentences):
        print(f"Error: Sentences file not found: {args.sentences}")
        sys.exit(1)

    if not os.path.exists(args.reference):
        print(f"Error: Reference audio file not found: {args.reference}")
        sys.exit(1)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    # Load sentences
    print(f"\nLoading sentences from: {args.sentences}")
    sentences = load_sentences(args.sentences)
    print(f"Found {len(sentences)} sentences to process")

    if len(sentences) == 0:
        print("Error: No sentences found in file")
        sys.exit(1)

    # Setup device
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")

    # Load model
    print("\n" + "-" * 70)
    print("[1/3] Loading model...")
    print("-" * 70)
    model, tokenizer = load_model(args.checkpoint, args.hf_token)
    model.to(device)
    model.eval()

    # Setup audio codec
    print("\n" + "-" * 70)
    print("[2/3] Setting up audio codec...")
    print("-" * 70)
    tts_codec = TTSCodec()
    tts_codec = setup_custom_encode(tts_codec)

    # Print generation parameters
    print("\nGeneration parameters:")
    print(f"  Temperature: {args.temperature}")
    print(f"  Top-k: {args.top_k}")
    print(f"  Top-p: {args.top_p}")
    print(f"  Repetition penalty: {args.repetition_penalty}")
    print(f"  Max audio tokens: {args.max_tokens}")
    print(f"  Output sample rate: {args.sample_rate}")

    # Process sentences
    print("\n" + "-" * 70)
    print("[3/3] Generating audio files...")
    print("-" * 70)

    successful = 0
    failed = 0
    failed_sentences = []

    for i, sentence in enumerate(sentences):
        index = args.start_index + i
        output_filename = f"{args.prefix}_{index:04d}.wav"
        output_path = output_dir / output_filename

        print(f"\n[{i + 1}/{len(sentences)}] Processing...")
        print(f"  Text: {sentence[:80]}{'...' if len(sentence) > 80 else ''}")
        print(f"  Output: {output_filename}")

        try:
            audio = infer(
                model=model,
                tokenizer=tokenizer,
                tts_codec=tts_codec,
                text=sentence,
                audio_file=args.reference,
                top_k=args.top_k,
                top_p=args.top_p,
                temperature=args.temperature,
                repetition_penalty=args.repetition_penalty,
                max_new_audio_tokens=args.max_tokens,
                device=device
            )
            save_audio(audio, str(output_path), args.sample_rate)
            successful += 1
            print(f"  Status: Success")
        except Exception as e:
            failed += 1
            failed_sentences.append((index, sentence, str(e)))
            print(f"  Status: FAILED - {e}")

    # Summary
    print("\n" + "=" * 70)
    print("Batch Inference Complete!")
    print("=" * 70)
    print(f"\nSummary:")
    print(f"  Total sentences: {len(sentences)}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Output directory: {output_dir}")

    if failed_sentences:
        print(f"\nFailed sentences:")
        for idx, sent, err in failed_sentences:
            print(f"  [{idx}] {sent[:50]}... - Error: {err}")

    # Save metadata
    metadata_path = output_dir / "metadata.txt"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        f.write("# MiraTTS Batch Inference Metadata\n")
        f.write(f"# Checkpoint: {args.checkpoint}\n")
        f.write(f"# Reference audio: {args.reference}\n")
        f.write(f"# Total: {len(sentences)}, Success: {successful}, Failed: {failed}\n")
        f.write("#\n")
        f.write("# Format: filename|text\n")
        f.write("#\n")
        for i, sentence in enumerate(sentences):
            index = args.start_index + i
            output_filename = f"{args.prefix}_{index:04d}.wav"
            f.write(f"{output_filename}|{sentence}\n")

    print(f"\nMetadata saved to: {metadata_path}")


if __name__ == "__main__":
    main()
