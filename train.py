#!/usr/bin/env python3
"""
Training script for MiraTTS using configuration file
Supports both LJSpeech and custom datasets with metadata files
"""
import argparse
import yaml
import os
from pathlib import Path
from src.mira_tts import (
    Config, ModelLoader, DataProcessor, MiraTrainer,
    LJSpeechLoader, CustomDatasetLoader
)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Train MiraTTS model using configuration file"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--override-output-dir",
        type=str,
        default=None,
        help="Override output directory from config"
    )
    parser.add_argument(
        "--override-num-samples",
        type=int,
        default=None,
        help="Override number of samples from config"
    )
    parser.add_argument(
        "--override-max-steps",
        type=int,
        default=None,
        help="Override max steps from config"
    )

    return parser.parse_args()


def main():
    """Main training function"""
    args = parse_args()

    # Load configuration
    print("=" * 80)
    print("MiraTTS Finetuning with Configuration File")
    print("=" * 80)
    print(f"Loading configuration from: {args.config}")

    cfg = load_config(args.config)

    # Initialize base configuration
    config = Config()

    # Apply configuration from file
    dataset_cfg = cfg.get('dataset', {})
    training_cfg = cfg.get('training', {})
    model_cfg = cfg.get('model', {})
    logging_cfg = cfg.get('logging', {})
    hub_cfg = cfg.get('hub', {})

    # Dataset configuration
    dataset_type = dataset_cfg.get('type', 'ljspeech')
    dataset_path = dataset_cfg.get('path')
    num_samples = dataset_cfg.get('num_samples')

    # Training configuration
    config.MAX_STEPS = training_cfg.get('max_steps', 60)
    config.LEARNING_RATE = training_cfg.get('learning_rate', 2e-4)
    config.PER_DEVICE_TRAIN_BATCH_SIZE = training_cfg.get('batch_size', 2)

    # Model configuration
    config.MODEL_NAME = model_cfg.get('pretrained_model', 'YatharthS/MiraTTS')
    config.OUTPUT_DIR = model_cfg.get('output_dir', 'outputs')

    # Logging configuration
    config.USE_WANDB = logging_cfg.get('use_wandb', False)
    config.WANDB_PROJECT = logging_cfg.get('wandb_project', 'miratts-finetuning')
    config.WANDB_RUN_NAME = logging_cfg.get('wandb_run_name')
    config.WANDB_ENTITY = logging_cfg.get('wandb_entity')
    config.WANDB_API_KEY = logging_cfg.get('wandb_api_key')
    config.USE_TENSORBOARD = logging_cfg.get('use_tensorboard', False)
    config.TENSORBOARD_LOG_DIR = logging_cfg.get('tensorboard_log_dir', 'runs')

    # Override with environment variable if available
    if 'WANDB_API_KEY' in os.environ and not config.WANDB_API_KEY:
        config.WANDB_API_KEY = os.environ['WANDB_API_KEY']

    # HuggingFace Hub configuration
    push_to_hub = hub_cfg.get('push_to_hub', False)
    if hub_cfg.get('repo_name'):
        config.UPLOAD_MODEL_REPO = hub_cfg['repo_name']
    if hub_cfg.get('token'):
        config.HF_TOKEN = hub_cfg['token']
    elif 'HF_TOKEN' in os.environ:
        config.HF_TOKEN = os.environ['HF_TOKEN']

    # Apply command line overrides
    if args.override_output_dir:
        config.OUTPUT_DIR = args.override_output_dir
    if args.override_num_samples is not None:
        num_samples = args.override_num_samples
        config.NUM_SAMPLES = num_samples
    if args.override_max_steps is not None:
        config.MAX_STEPS = args.override_max_steps

    # Print configuration summary
    print("\nConfiguration Summary:")
    print(f"  Dataset type: {dataset_type}")
    print(f"  Dataset path: {dataset_path}")
    print(f"  Number of samples: {num_samples if num_samples else 'all'}")
    print(f"  Max steps: {config.MAX_STEPS}")
    print(f"  Learning rate: {config.LEARNING_RATE}")
    print(f"  Batch size: {config.PER_DEVICE_TRAIN_BATCH_SIZE}")
    print(f"  Output directory: {config.OUTPUT_DIR}")
    print(f"  WandB enabled: {config.USE_WANDB}")
    if config.USE_WANDB:
        print(f"  WandB project: {config.WANDB_PROJECT}")
    print(f"  TensorBoard enabled: {config.USE_TENSORBOARD}")
    print(f"  Push to Hub: {push_to_hub}")
    print("=" * 80)

    # Step 1: Load dataset
    print("\n[1/5] Loading dataset...")

    if dataset_type == 'ljspeech':
        loader = LJSpeechLoader(dataset_path)
        raw_dataset = loader.load_dataset(num_samples=num_samples)
    elif dataset_type == 'custom':
        metadata_file = dataset_cfg.get('metadata_file', 'metadata.csv')
        loader = CustomDatasetLoader(dataset_path, metadata_file)
        raw_dataset = loader.load_dataset(num_samples=num_samples)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}. Must be 'ljspeech' or 'custom'")

    print(f"Loaded {len(raw_dataset)} samples")

    # Step 2: Load model
    print("\n[2/5] Loading model...")
    model_loader = ModelLoader(config)
    model, tokenizer = model_loader.load_model()
    model_loader.get_gpu_stats()

    # Step 3: Process dataset
    print("\n[3/5] Processing audio data...")
    data_processor = DataProcessor(config)
    data_processor.dataset = raw_dataset

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
    if push_to_hub:
        print("\nPushing model to HuggingFace Hub...")
        trainer.push_to_hub()

    print("\n" + "=" * 80)
    print("Training pipeline completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
