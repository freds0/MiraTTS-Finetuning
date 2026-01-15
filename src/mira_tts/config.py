"""
Configuration file for MiraTTS training
"""
import os

class Config:
    """Configuration class for MiraTTS training"""

    # Model settings
    MODEL_NAME = "YatharthS/MiraTTS"
    MAX_SEQ_LENGTH = 30 * 50  # 30 seconds of audio
    DTYPE = "float32"
    LOAD_IN_4BIT = False
    FULL_FINETUNING = True

    # Dataset settings
    DATASET_NAME = "WpythonW/elevenlabs_multilingual_v2-technical-speech"
    DATASET_SPLIT = "test"
    TEXT_COLUMN = "text"
    AUDIO_COLUMN = "audio"
    NUM_SAMPLES = 20  # Number of samples to train on
    SAMPLING_RATE = 16000
    AUDIO_DURATION = 30.0  # Maximum audio duration in seconds

    # Training settings
    PER_DEVICE_TRAIN_BATCH_SIZE = 2
    GRADIENT_ACCUMULATION_STEPS = 4
    WARMUP_STEPS = 5
    MAX_STEPS = 60
    LEARNING_RATE = 2e-4
    FP16 = False  # Full float32
    BF16 = False  # Full float32
    LOGGING_STEPS = 1
    OPTIM = "adamw_8bit"
    WEIGHT_DECAY = 0.001
    LR_SCHEDULER_TYPE = "linear"
    SEED = 3407
    OUTPUT_DIR = "outputs"
    REPORT_TO = "none"  # Can be "wandb", "tensorboard", etc.
    PACKING = False

    # Inference settings
    INFERENCE_TOP_K = 50
    INFERENCE_TOP_P = 1.0
    INFERENCE_TEMPERATURE = 0.8
    INFERENCE_REPETITION_PENALTY = 1.2
    MAX_NEW_AUDIO_TOKENS = 1024
    OUTPUT_SAMPLE_RATE = 48000

    # Upload settings
    UPLOAD_MODEL_REPO = "username/my-awesome-finetuned-model"
    HF_TOKEN = ""  # Set your Hugging Face token here

    # Environment settings
    @staticmethod
    def setup_environment():
        """Setup environment variables"""
        os.environ['UNSLOTH_FORCE_FLOAT32'] = '1'
        os.environ['ACCELERATE_MIXED_PRECISION'] = 'no'
