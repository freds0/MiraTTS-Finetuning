"""
LJSpeech dataset loader for local files
"""
import os
import csv
from datasets import Dataset, Audio, Features, Value
from pathlib import Path


class LJSpeechLoader:
    """Class for loading LJSpeech dataset from local directory"""

    def __init__(self, dataset_path: str):
        """
        Initialize LJSpeechLoader

        Args:
            dataset_path: Path to LJSpeech dataset directory
        """
        self.dataset_path = Path(dataset_path)
        self.metadata_file = self.dataset_path / "metadata.csv"
        self.wavs_dir = self.dataset_path / "wavs"

        if not self.dataset_path.exists():
            raise ValueError(f"Dataset path not found: {dataset_path}")
        if not self.metadata_file.exists():
            raise ValueError(f"Metadata file not found: {self.metadata_file}")
        if not self.wavs_dir.exists():
            raise ValueError(f"Wavs directory not found: {self.wavs_dir}")

    def load_dataset(self, split: str = "train", num_samples: int = None):
        """
        Load LJSpeech dataset

        Args:
            split: Dataset split ("train", "test", or "all")
            num_samples: Number of samples to load. If None, loads all.

        Returns:
            Dataset: HuggingFace Dataset object
        """
        print(f"Loading LJSpeech dataset from {self.dataset_path}...")

        # Read metadata
        data = []
        with open(self.metadata_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='|')
            for i, row in enumerate(reader):
                if num_samples and i >= num_samples:
                    break

                file_id = row[0]
                text = row[2] if len(row) > 2 else row[1]  # Use normalized text if available
                audio_path = str(self.wavs_dir / f"{file_id}.wav")

                # Check if audio file exists
                if os.path.exists(audio_path):
                    data.append({
                        "audio": audio_path,
                        "text": text,
                        "file_id": file_id
                    })

        print(f"Loaded {len(data)} samples from LJSpeech")

        # Create dataset
        dataset = Dataset.from_list(data)

        # Cast audio column to Audio feature
        dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

        return dataset

    def get_train_test_split(self, train_size: float = 0.9, num_samples: int = None):
        """
        Get train/test split from LJSpeech dataset

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
    def create_from_path(dataset_path: str):
        """
        Factory method to create LJSpeechLoader

        Args:
            dataset_path: Path to LJSpeech dataset directory

        Returns:
            LJSpeechLoader: Loader instance
        """
        return LJSpeechLoader(dataset_path)
