"""
Audio codec module for encoding and decoding audio
"""
import torch
import numpy as np
from ncodec.codec import TTSCodec
from ncodec.encoder.model import audio_volume_normalize


class AudioCodecManager:
    """Class for managing audio encoding and decoding"""

    def __init__(self):
        """Initialize the audio codec"""
        self.tts_codec = TTSCodec()
        self._setup_codec()

    def _setup_codec(self):
        """Setup the audio codec with custom encode function"""
        # Patch the encode method
        self.tts_codec.audio_encoder.encode = self._custom_encode
        self.tts_codec.audio_encoder.feature_extractor.config.output_hidden_states = True

    @torch.inference_mode()
    def _custom_encode(self, audio, encode_semantic=True, duration=8):
        """
        Encodes audio file into speech tokens and context tokens

        Args:
            audio: Audio array
            encode_semantic: Whether to encode semantic tokens
            duration: Duration of audio in seconds

        Returns:
            tuple: (speech_tokens, context_tokens) or just context_tokens
        """
        audio_encoder = self.tts_codec.audio_encoder

        # Normalize audio volume
        audio = audio_volume_normalize(audio)

        # Get reference clip
        ref_clip = audio_encoder.get_ref_clip(audio)
        wav_ref = torch.from_numpy(ref_clip).unsqueeze(0).float()

        # Extract mel spectrogram
        mel = audio_encoder.mel_transformer(wav_ref).squeeze(1)
        new_arr = np.array(mel.transpose(1, 2).cpu())

        # Generate global tokens
        global_tokens = audio_encoder.s_encoder.run(
            ["global_tokens"],
            {"mel_spectrogram": new_arr}
        )
        context_tokens = "".join([
            f"<|context_token_{i}|>" for i in global_tokens[0].squeeze()
        ])

        if encode_semantic:
            # Extract semantic tokens
            feat = audio_encoder.extract_wav2vec2_features(audio)
            speech_tokens = audio_encoder.q_encoder.run(
                ["semantic_tokens"],
                {"features": feat.cpu().detach().numpy()}
            )
            speech_tokens = "".join([
                f"<|speech_token_{i}|>" for i in speech_tokens[0][0]
            ])
            return speech_tokens, context_tokens
        else:
            return context_tokens

    def encode(self, audio_array, encode_semantic=True, duration=30.0):
        """
        Encode audio to tokens

        Args:
            audio_array: Audio array (numpy array)
            encode_semantic: Whether to encode semantic tokens
            duration: Duration in seconds

        Returns:
            tuple: (speech_tokens, context_tokens) or just context_tokens
        """
        return self.tts_codec.audio_encoder.encode(
            audio_array,
            encode_semantic,
            duration
        )

    def decode(self, predict_text, context_tokens):
        """
        Decode tokens to audio

        Args:
            predict_text: Predicted text tokens
            context_tokens: Context tokens

        Returns:
            numpy.ndarray: Audio array
        """
        return self.tts_codec.decode(predict_text, context_tokens)

    def format_prompt(self, text, context_tokens, additional_context=None):
        """
        Format the prompt for TTS generation

        Args:
            text: Text to synthesize
            context_tokens: Context tokens from reference audio
            additional_context: Additional context (optional)

        Returns:
            str: Formatted prompt
        """
        return self.tts_codec.format_prompt(text, context_tokens, additional_context)
