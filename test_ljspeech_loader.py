#!/usr/bin/env python3
"""
Test script to verify LJSpeech dataset loading
"""
import sys
sys.path.insert(0, '/home/fred/Projetos/Mira-TTS-Finetuning')

from src.mira_tts.ljspeech_loader import LJSpeechLoader


def main():
    """Test LJSpeech dataset loading"""
    print("=" * 80)
    print("Testing LJSpeech Dataset Loader")
    print("=" * 80)

    dataset_path = "/home/fred/Projetos/DATASETS/LJSpeech-1.1/"
    num_samples = 10

    print(f"\nDataset path: {dataset_path}")
    print(f"Number of samples to load: {num_samples}\n")

    # Create loader
    print("[1/3] Creating LJSpeech loader...")
    loader = LJSpeechLoader(dataset_path)
    print("Loader created successfully!")

    # Load dataset
    print("\n[2/3] Loading dataset...")
    dataset = loader.load_dataset(num_samples=num_samples)
    print(f"Dataset loaded successfully!")
    print(f"Total samples: {len(dataset)}")

    # Display sample data
    print("\n[3/3] Displaying sample data...")
    print("-" * 80)

    for i in range(min(3, len(dataset))):
        sample = dataset[i]
        print(f"\nSample {i+1}:")
        print(f"  File ID: {sample.get('file_id', 'N/A')}")
        print(f"  Text: {sample['text'][:100]}...")
        print(f"  Audio path: {sample['audio']['path']}")
        print(f"  Sample rate: {sample['audio']['sampling_rate']}")
        print(f"  Audio array shape: {sample['audio']['array'].shape}")

    print("\n" + "=" * 80)
    print("LJSpeech dataset loading test completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
