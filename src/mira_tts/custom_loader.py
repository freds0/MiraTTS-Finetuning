"""
Custom dataset loader for datasets with metadata file
"""
import os
from datasets import Dataset, Audio
from pathlib import Path


class CustomDatasetLoader:
    """Class for loading custom dataset from local directory with metadata file"""

    def __init__(self, dataset_path: str, metadata_file: str = "metadata.csv"):
        """
        Initialize CustomDatasetLoader

        Args:
            dataset_path: Path to dataset directory (root folder)
            metadata_file: Name of the metadata file (relative to dataset_path)
                          Format: "filepath|text" (one entry per line)
                          filepath can be absolute or relative to dataset_path
        """
        self.dataset_path = Path(dataset_path)
        self.metadata_file = self.dataset_path / metadata_file

        if not self.dataset_path.exists():
            raise ValueError(f"Dataset path not found: {dataset_path}")
        if not self.metadata_file.exists():
            raise ValueError(f"Metadata file not found: {self.metadata_file}")

    def load_dataset(self, split: str = "train", num_samples: int = None):
        """
        Load custom dataset from metadata file

        Args:
            split: Dataset split ("train", "test", or "all")
            num_samples: Number of samples to load. If None, loads all.

        Returns:
            Dataset: HuggingFace Dataset object
        """
        print(f"Loading custom dataset from {self.dataset_path}...")
        print(f"Metadata file: {self.metadata_file}")

        # Read metadata
        data = []
        skipped = 0

        with open(self.metadata_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if num_samples and i >= num_samples:
                    break

                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                # Parse format: "filepath|text"
                parts = line.split('|', 1)
                if len(parts) != 2:
                    print(f"Warning: Skipping malformed line {i+1}: {line[:50]}...")
                    skipped += 1
                    continue

                audio_path_str, text = parts
                audio_path_str = audio_path_str.strip()
                text = text.strip()

                # Handle both absolute and relative paths
                audio_path = Path(audio_path_str)
                if not audio_path.is_absolute():
                    audio_path = self.dataset_path / audio_path_str

                # Check if audio file exists
                if audio_path.exists():
                    data.append({
                        "audio": str(audio_path),
                        "text": text,
                        "file_id": audio_path.stem
                    })
                else:
                    print(f"Warning: Audio file not found: {audio_path}")
                    skipped += 1

        print(f"Loaded {len(data)} samples from custom dataset")
        if skipped > 0:
            print(f"Skipped {skipped} entries (missing files or malformed lines)")

        if len(data) == 0:
            raise ValueError("No valid samples found in metadata file")

        # Create dataset
        dataset = Dataset.from_list(data)

        # Cast audio column to Audio feature
        dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

        return dataset

    def get_train_test_split(self, train_size: float = 0.9, num_samples: int = None):
        """
        Get train/test split from custom dataset

        Args:
            train_size: Fraction of data to use for training
            num_samples: Total number of samples to load. If None, loads all.

        Returns:
            tuple: (train_dataset, test_dataset)
        """
        # Load full dataset
        full_dataset = self.load_dataset(split="all", num_samples=num_samples)

        # Split into train/test
        split_dataset = full_dataset.train_test_split(
            train_size=train_size,
            seed=42
        )

        return split_dataset["train"], split_dataset["test"]

    @staticmethod
    def create_from_path(dataset_path: str, metadata_file: str = "metadata.csv"):
        """
        Factory method to create CustomDatasetLoader

        Args:
            dataset_path: Path to dataset directory
            metadata_file: Name of the metadata file

        Returns:
            CustomDatasetLoader: Loader instance
        """
        return CustomDatasetLoader(dataset_path, metadata_file)
