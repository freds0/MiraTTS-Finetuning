#!/usr/bin/env python3
"""
Main training script for MiraTTS finetuning
"""
import argparse
from src.mira_tts import Config, ModelLoader, DataProcessor, MiraTrainer


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Train MiraTTS model with custom dataset"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="HuggingFace dataset name (default: from config)"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of samples to train on (default: from config)"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum training steps (default: from config)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Learning rate (default: from config)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for model (default: from config)"
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
        help="HuggingFace Hub repository name (default: from config)"
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="HuggingFace token for uploading (default: from config)"
    )

    return parser.parse_args()


def main():
    """Main training function"""
    args = parse_args()

    # Initialize configuration
    config = Config()

    # Override config with command line arguments
    if args.dataset:
        config.DATASET_NAME = args.dataset
    if args.num_samples:
        config.NUM_SAMPLES = args.num_samples
    if args.max_steps:
        config.MAX_STEPS = args.max_steps
    if args.learning_rate:
        config.LEARNING_RATE = args.learning_rate
    if args.output_dir:
        config.OUTPUT_DIR = args.output_dir
    if args.hub_repo:
        config.UPLOAD_MODEL_REPO = args.hub_repo
    if args.hf_token:
        config.HF_TOKEN = args.hf_token

    print("=" * 60)
    print("MiraTTS Finetuning")
    print("=" * 60)

    # Step 1: Load model
    print("\n[1/5] Loading model...")
    model_loader = ModelLoader(config)
    model, tokenizer = model_loader.load_model()
    model_loader.get_gpu_stats()

    # Step 2: Load and process dataset
    print("\n[2/5] Loading and processing dataset...")
    data_processor = DataProcessor(config)
    train_dataset = data_processor.process_dataset()

    # Step 3: Setup trainer
    print("\n[3/5] Setting up trainer...")
    trainer = MiraTrainer(model, tokenizer, config)
    trainer.setup_trainer(train_dataset)
    trainer.print_gpu_stats()

    # Step 4: Train
    print("\n[4/5] Training model...")
    trainer_stats = trainer.train()

    print("\nTraining completed!")
    print(f"Training statistics: {trainer_stats}")

    # Step 5: Save model
    print("\n[5/5] Saving model...")
    trainer.save_model()

    # Optional: Push to hub
    if args.push_to_hub:
        print("\nPushing model to HuggingFace Hub...")
        trainer.push_to_hub()

    print("\n" + "=" * 60)
    print("Training pipeline completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
