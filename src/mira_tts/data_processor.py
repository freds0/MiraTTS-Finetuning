"""
Data processing module for MiraTTS dataset
"""
from datasets import load_dataset, Audio
from .audio_codec import AudioCodecManager
from .config import Config


class DataProcessor:
    """Class for processing audio datasets"""

    def __init__(self, config: Config = None):
        """
        Initialize DataProcessor

        Args:
            config: Configuration object. If None, uses default Config.
        """
        self.config = config or Config()
        self.audio_codec = AudioCodecManager()
        self.dataset = None

    def load_dataset(self):
        """
        Load dataset from HuggingFace

        Returns:
            Dataset: Loaded dataset
        """
        print(f"Loading dataset {self.config.DATASET_NAME}...")
        self.dataset = load_dataset(
            self.config.DATASET_NAME,
            split=self.config.DATASET_SPLIT
        )

        # Rename columns to standard names
        self.dataset = self.dataset.rename_columns({
            self.config.TEXT_COLUMN: "text",
            self.config.AUDIO_COLUMN: "audio"
        })

        print(f"Dataset loaded with {len(self.dataset)} samples")
        return self.dataset

    def process_dataset(self, num_samples: int = None):
        """
        Process the dataset for training

        Args:
            num_samples: Number of samples to process. If None, uses config value.

        Returns:
            Dataset: Processed dataset
        """
        if self.dataset is None:
            self.load_dataset()

        num_samples = num_samples or self.config.NUM_SAMPLES

        print(f"Processing {num_samples} samples...")
        small_dataset = self.dataset.select(range(num_samples))
        small_dataset = small_dataset.cast_column(
            "audio",
            Audio(sampling_rate=self.config.SAMPLING_RATE)
        )

        # Process audio files
        processed_dataset = small_dataset.map(
            self._process_audio_sample,
            remove_columns=["audio"]
        )

        print("Dataset processing complete!")
        return processed_dataset

    def _process_audio_sample(self, example):
        """
        Process a single audio sample

        Args:
            example: Dataset example with audio and text

        Returns:
            dict: Processed example with formatted prompt
        """
        audio_array = example["audio"]["array"]
        text = example['text']

        print(f"Processing audio with shape: {audio_array.shape}")

        # Encode audio to tokens
        semantic_tokens, global_tokens = self.audio_codec.encode(
            audio_array,
            encode_semantic=True,
            duration=self.config.AUDIO_DURATION
        )

        # Format prompt
        prompt = (
            f"<|task_tts|><|start_text|>{text}<|end_text|>"
            f"<|context_audio_start|>{global_tokens}<|context_audio_end|>"
            f"<|prompt_speech_start|>{semantic_tokens}"
        )

        return {'text': prompt}

    def get_sample_info(self, index: int = 0):
        """
        Get information about a specific sample

        Args:
            index: Index of the sample

        Returns:
            dict: Sample information
        """
        if self.dataset is None:
            self.load_dataset()

        sample = self.dataset[index]
        return {
            "index": index,
            "text": sample.get("text", ""),
            "audio_info": sample.get("audio", {}),
        }
