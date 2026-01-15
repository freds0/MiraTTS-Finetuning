"""
Model loader module for MiraTTS
"""
import torch
from unsloth import FastModel
from huggingface_hub import snapshot_download
from .config import Config


class ModelLoader:
    """Class for loading and managing MiraTTS model"""

    def __init__(self, config: Config = None):
        """
        Initialize ModelLoader

        Args:
            config: Configuration object. If None, uses default Config.
        """
        self.config = config or Config()
        self.model = None
        self.tokenizer = None
        self.model_path = None

    def load_model(self):
        """
        Load the MiraTTS model and tokenizer

        Returns:
            tuple: (model, tokenizer)
        """
        Config.setup_environment()

        print(f"Downloading model from {self.config.MODEL_NAME}...")
        self.model_path = snapshot_download(self.config.MODEL_NAME)

        print(f"Loading model...")
        self.model, self.tokenizer = FastModel.from_pretrained(
            model_name=self.config.MODEL_NAME,
            max_seq_length=self.config.MAX_SEQ_LENGTH,
            dtype=getattr(torch, self.config.DTYPE),
            full_finetuning=self.config.FULL_FINETUNING,
            load_in_4bit=self.config.LOAD_IN_4BIT,
            torch_dtype=self.config.DTYPE,
        )

        print("Model loaded successfully!")
        return self.model, self.tokenizer

    def get_gpu_stats(self):
        """
        Get GPU memory statistics

        Returns:
            dict: Dictionary with GPU statistics
        """
        if not torch.cuda.is_available():
            return {"error": "CUDA not available"}

        gpu_stats = torch.cuda.get_device_properties(0)
        start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)

        stats = {
            "gpu_name": gpu_stats.name,
            "max_memory_gb": max_memory,
            "reserved_memory_gb": start_gpu_memory,
        }

        print(f"GPU = {stats['gpu_name']}. Max memory = {stats['max_memory_gb']} GB.")
        print(f"{stats['reserved_memory_gb']} GB of memory reserved.")

        return stats
