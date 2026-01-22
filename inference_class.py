#!/usr/bin/env python3
"""
Batch inference script for MiraTTS using MiraInference class

Generates audio files from a list of sentences using a trained checkpoint.
"""
import argparse
import os
import sys
from pathlib import Path

from src.mira_tts import Config, ModelLoader, MiraInference


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Batch inference for MiraTTS - Generate audio from text sentences",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with local checkpoint
  python inference_class.py \\
      --checkpoint ./checkpoints/ \\
      --sentences sentences.txt \\
      --reference reference.wav \\
      --output-dir ./generated_audio/

  # With HuggingFace model
  python inference_class.py \\
      --checkpoint YatharthS/MiraTTS \\
      --sentences sentences.txt \\
      --reference reference.wav \\
      --output-dir ./generated_audio/ \\
      --hf-token your_token_here

  # With custom generation parameters
  python inference_class.py \\
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
        default=None,
        help="Sampling temperature (default: 0.8)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Top-k sampling (default: 50)"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Top-p (nucleus) sampling (default: 1.0)"
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=None,
        help="Repetition penalty (default: 1.2)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
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

    return parser.parse_args()


def load_sentences(file_path: str) -> list:
    """Load sentences from a text file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        sentences = [line.strip() for line in f if line.strip()]
    return sentences


def main():
    """Main batch inference function"""
    args = parse_args()

    print("=" * 70)
    print("MiraTTS Batch Inference (using MiraInference class)")
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

    # Initialize configuration
    config = Config()
    config.MODEL_NAME = args.checkpoint

    # Load model
    print("\n" + "-" * 70)
    print("[1/3] Loading model...")
    print("-" * 70)
    model_loader = ModelLoader(config)
    model, tokenizer = model_loader.load_model(hf_token=args.hf_token)

    # Setup inference engine
    print("\n" + "-" * 70)
    print("[2/3] Setting up inference engine...")
    print("-" * 70)
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

    # Print generation parameters
    print("\nGeneration parameters:")
    print(f"  Temperature: {gen_kwargs.get('temperature', config.INFERENCE_TEMPERATURE)}")
    print(f"  Top-k: {gen_kwargs.get('top_k', config.INFERENCE_TOP_K)}")
    print(f"  Top-p: {gen_kwargs.get('top_p', config.INFERENCE_TOP_P)}")
    print(f"  Repetition penalty: {gen_kwargs.get('repetition_penalty', config.INFERENCE_REPETITION_PENALTY)}")
    print(f"  Max audio tokens: {gen_kwargs.get('max_new_audio_tokens', config.MAX_NEW_AUDIO_TOKENS)}")
    print(f"  Output sample rate: {config.OUTPUT_SAMPLE_RATE}")

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
            inference.infer_and_save(
                text=sentence,
                audio_file=args.reference,
                output_path=str(output_path),
                **gen_kwargs
            )
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
