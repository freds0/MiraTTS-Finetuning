#!/usr/bin/env python3
"""
Test script for MiraTTS inference
"""
import argparse
import os
from src.mira_tts import Config, ModelLoader, MiraInference


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Test MiraTTS model inference"
    )
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="Text to synthesize"
    )
    parser.add_argument(
        "--audio-file",
        type=str,
        required=True,
        help="Reference audio file path"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output.wav",
        help="Output audio file path (default: output.wav)"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to trained model (default: from config)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature (default: from config)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Top-k sampling (default: from config)"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Top-p (nucleus) sampling (default: from config)"
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=None,
        help="Repetition penalty (default: from config)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Maximum new audio tokens (default: from config)"
    )

    return parser.parse_args()


def main():
    """Main inference function"""
    args = parse_args()

    # Initialize configuration
    config = Config()

    # Override model name if model path is provided
    if args.model_path:
        config.MODEL_NAME = args.model_path

    print("=" * 60)
    print("MiraTTS Inference Test")
    print("=" * 60)

    # Check if reference audio file exists
    if not os.path.exists(args.audio_file):
        print(f"Error: Audio file not found: {args.audio_file}")
        return

    # Load model
    print("\n[1/3] Loading model...")
    model_loader = ModelLoader(config)
    model, tokenizer = model_loader.load_model()

    # Setup inference
    print("\n[2/3] Setting up inference...")
    inference = MiraInference(model, tokenizer, config)

    # Prepare generation parameters
    gen_kwargs = {}
    if args.temperature is not None:
        gen_kwargs['temperature'] = args.temperature
    if args.top_k is not None:
        gen_kwargs['top_k'] = args.top_k
    if args.top_p is not None:
        gen_kwargs['top_p'] = args.top_p
    if args.repetition_penalty is not None:
        gen_kwargs['repetition_penalty'] = args.repetition_penalty
    if args.max_tokens is not None:
        gen_kwargs['max_new_audio_tokens'] = args.max_tokens

    # Run inference
    print("\n[3/3] Generating speech...")
    print(f"Text: {args.text}")
    print(f"Reference audio: {args.audio_file}")

    audio = inference.infer_and_save(
        text=args.text,
        audio_file=args.audio_file,
        output_path=args.output,
        **gen_kwargs
    )

    print("\n" + "=" * 60)
    print(f"Inference completed successfully!")
    print(f"Output saved to: {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
