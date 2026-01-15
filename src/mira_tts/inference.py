"""
Inference module for MiraTTS
"""
import torch
import librosa
from .audio_codec import AudioCodecManager
from .config import Config


class MiraInference:
    """Class for running inference with MiraTTS model"""

    def __init__(self, model, tokenizer, config: Config = None, device: str = None):
        """
        Initialize MiraInference

        Args:
            model: The trained model
            tokenizer: The tokenizer
            config: Configuration object. If None, uses default Config.
            device: Device to run inference on. If None, uses cuda:0 if available.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or Config()
        self.device = device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.audio_codec = AudioCodecManager()

        # Move model to device
        self.model.to(self.device)
        self.model.eval()

    def infer(
        self,
        text: str,
        audio_file: str,
        top_k: int = None,
        top_p: float = None,
        temperature: float = None,
        repetition_penalty: float = None,
        max_new_audio_tokens: int = None
    ):
        """
        Generate speech from text using reference audio

        Args:
            text: Text to synthesize
            audio_file: Path to reference audio file
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            temperature: Sampling temperature
            repetition_penalty: Repetition penalty
            max_new_audio_tokens: Maximum number of new audio tokens to generate

        Returns:
            numpy.ndarray: Generated audio array
        """
        # Use config defaults if not specified
        top_k = top_k or self.config.INFERENCE_TOP_K
        top_p = top_p or self.config.INFERENCE_TOP_P
        temperature = temperature or self.config.INFERENCE_TEMPERATURE
        repetition_penalty = repetition_penalty or self.config.INFERENCE_REPETITION_PENALTY
        max_new_audio_tokens = max_new_audio_tokens or self.config.MAX_NEW_AUDIO_TOKENS

        # Load and process reference audio
        print(f"Loading reference audio from {audio_file}...")
        audio_array, sr = librosa.load(audio_file, sr=self.config.SAMPLING_RATE)

        # Encode reference audio
        print("Encoding reference audio...")
        context_tokens = self.audio_codec.encode(audio_array, encode_semantic=False)

        # Format prompt
        formatted_prompt = self.audio_codec.format_prompt(text, context_tokens, None)

        # Tokenize
        model_inputs = self.tokenizer([formatted_prompt], return_tensors="pt").to(self.device)

        # Generate
        print("Generating token sequence...")
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_audio_tokens,
                do_sample=True,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id
            )
        print("Token sequence generated.")

        # Decode tokens
        generated_ids_trimmed = generated_ids[:, model_inputs.input_ids.shape[1]:]
        predicts_text = self.tokenizer.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=False
        )[0]

        # Decode audio
        print("Decoding audio...")
        audio = self.audio_codec.decode(predicts_text, context_tokens)

        return audio

    def save_audio(self, audio, output_path: str, sample_rate: int = None):
        """
        Save audio to file

        Args:
            audio: Audio array
            output_path: Path to save audio file
            sample_rate: Sample rate. If None, uses config value.
        """
        import soundfile as sf

        sample_rate = sample_rate or self.config.OUTPUT_SAMPLE_RATE
        print(f"Saving audio to {output_path}...")
        sf.write(output_path, audio, sample_rate)
        print("Audio saved!")

    def infer_and_save(
        self,
        text: str,
        audio_file: str,
        output_path: str,
        **generation_kwargs
    ):
        """
        Generate speech and save to file

        Args:
            text: Text to synthesize
            audio_file: Path to reference audio file
            output_path: Path to save generated audio
            **generation_kwargs: Additional generation parameters

        Returns:
            numpy.ndarray: Generated audio array
        """
        audio = self.infer(text, audio_file, **generation_kwargs)
        self.save_audio(audio, output_path)
        return audio
