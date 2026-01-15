#!/usr/bin/env python3
"""
Simple test script to verify LJSpeech dataset loading (without model dependencies)
"""
import os
import csv
from pathlib import Path
from datasets import Dataset, Audio


def load_ljspeech(dataset_path, num_samples=10):
    """Load LJSpeech dataset from local directory"""
    dataset_path = Path(dataset_path)
    metadata_file = dataset_path / "metadata.csv"
    wavs_dir = dataset_path / "wavs"

    if not dataset_path.exists():
        raise ValueError(f"Dataset path not found: {dataset_path}")
    if not metadata_file.exists():
        raise ValueError(f"Metadata file not found: {metadata_file}")
    if not wavs_dir.exists():
        raise ValueError(f"Wavs directory not found: {wavs_dir}")

    print(f"Loading LJSpeech dataset from {dataset_path}...")

    # Read metadata
    data = []
    with open(metadata_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='|')
        for i, row in enumerate(reader):
            if i >= num_samples:
                break

            file_id = row[0]
            text = row[2] if len(row) > 2 else row[1]
            audio_path = str(wavs_dir / f"{file_id}.wav")

            if os.path.exists(audio_path):
                data.append({
                    "audio": audio_path,
                    "text": text,
                    "file_id": file_id
                })

    print(f"Loaded {len(data)} samples from LJSpeech")

    # Create dataset
    dataset = Dataset.from_list(data)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    return dataset


def main():
    """Test LJSpeech dataset loading"""
    print("=" * 80)
    print("Testing LJSpeech Dataset Loader (Simple Version)")
    print("=" * 80)

    dataset_path = "/home/fred/Projetos/DATASETS/LJSpeech-1.1/"
    num_samples = 10

    print(f"\nDataset path: {dataset_path}")
    print(f"Number of samples to load: {num_samples}\n")

    # Load dataset
    print("[1/2] Loading dataset...")
    dataset = load_ljspeech(dataset_path, num_samples=num_samples)
    print(f"Dataset loaded successfully!")
    print(f"Total samples: {len(dataset)}")

    # Display sample data
    print("\n[2/2] Displaying sample data...")
    print("-" * 80)

    for i in range(min(3, len(dataset))):
        sample = dataset[i]
        print(f"\nSample {i+1}:")
        print(f"  File ID: {sample.get('file_id', 'N/A')}")
        print(f"  Text: {sample['text'][:100]}...")
        print(f"  Audio path: {sample['audio']['path']}")
        print(f"  Sample rate: {sample['audio']['sampling_rate']}")
        print(f"  Audio array shape: {sample['audio']['array'].shape}")
        print(f"  Audio duration: {len(sample['audio']['array']) / sample['audio']['sampling_rate']:.2f}s")

    print("\n" + "=" * 80)
    print("LJSpeech dataset loading test completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
