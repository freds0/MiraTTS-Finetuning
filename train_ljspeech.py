#!/usr/bin/env python3
"""
Training script for MiraTTS using LJSpeech local dataset
"""
import argparse
from src.mira_tts import Config, ModelLoader, DataProcessor, MiraTrainer, LJSpeechLoader


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Train MiraTTS model with LJSpeech dataset"
    )
    parser.add_argument(
        "--ljspeech-path",
        type=str,
        default="/home/fred/Projetos/DATASETS/LJSpeech-1.1/",
        help="Path to LJSpeech dataset directory"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=20,
        help="Number of samples to train on (default: 20)"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=60,
        help="Maximum training steps (default: 60)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Learning rate (default: 2e-4)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size per device (default: 2)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs_ljspeech",
        help="Output directory for model (default: outputs_ljspeech)"
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push model to HuggingFace Hub after training"
    )
    parser.add_argument(
        "--hub-repo",
        type=str,
        default=None,
        help="HuggingFace Hub repository name"
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="HuggingFace token for uploading"
    )

    return parser.parse_args()


def main():
    """Main training function"""
    args = parse_args()

    # Initialize configuration
    config = Config()

    # Override config with command line arguments
    config.NUM_SAMPLES = args.num_samples
    config.MAX_STEPS = args.max_steps
    config.LEARNING_RATE = args.learning_rate
    config.PER_DEVICE_TRAIN_BATCH_SIZE = args.batch_size
    config.OUTPUT_DIR = args.output_dir

    if args.hub_repo:
        config.UPLOAD_MODEL_REPO = args.hub_repo
    if args.hf_token:
        config.HF_TOKEN = args.hf_token

    print("=" * 80)
    print("MiraTTS Finetuning with LJSpeech Dataset")
    print("=" * 80)
    print(f"Dataset path: {args.ljspeech_path}")
    print(f"Number of samples: {config.NUM_SAMPLES}")
    print(f"Max steps: {config.MAX_STEPS}")
    print(f"Learning rate: {config.LEARNING_RATE}")
    print(f"Batch size: {config.PER_DEVICE_TRAIN_BATCH_SIZE}")
    print(f"Output directory: {config.OUTPUT_DIR}")
    print("=" * 80)

    # Step 1: Load LJSpeech dataset
    print("\n[1/5] Loading LJSpeech dataset...")
    ljspeech_loader = LJSpeechLoader(args.ljspeech_path)
    raw_dataset = ljspeech_loader.load_dataset(num_samples=config.NUM_SAMPLES)
    print(f"Loaded {len(raw_dataset)} samples")

    # Step 2: Load model
    print("\n[2/5] Loading model...")
    model_loader = ModelLoader(config)
    model, tokenizer = model_loader.load_model()
    model_loader.get_gpu_stats()

    # Step 3: Process dataset
    print("\n[3/5] Processing audio data...")
    data_processor = DataProcessor(config)
    data_processor.dataset = raw_dataset  # Use LJSpeech dataset

    # Process audio samples
    processed_dataset = data_processor.dataset.map(
        data_processor._process_audio_sample,
        remove_columns=["audio", "file_id"]
    )

    print(f"Processed {len(processed_dataset)} samples")

    # Step 4: Setup trainer
    print("\n[4/5] Setting up trainer...")
    trainer = MiraTrainer(model, tokenizer, config)
    trainer.setup_trainer(processed_dataset)
    trainer.print_gpu_stats()

    # Step 5: Train
    print("\n[5/5] Training model...")
    print("=" * 80)
    trainer_stats = trainer.train()

    print("\n" + "=" * 80)
    print("Training completed!")
    print(f"Training statistics:")
    print(f"  - Total steps: {trainer_stats.global_step if hasattr(trainer_stats, 'global_step') else 'N/A'}")
    print(f"  - Training loss: {trainer_stats.training_loss if hasattr(trainer_stats, 'training_loss') else 'N/A'}")
    print("=" * 80)

    # Save model
    print("\nSaving model...")
    trainer.save_model()
    print(f"Model saved to: {config.OUTPUT_DIR}")

    # Optional: Push to hub
    if args.push_to_hub:
        print("\nPushing model to HuggingFace Hub...")
        trainer.push_to_hub()

    print("\n" + "=" * 80)
    print("Training pipeline completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
