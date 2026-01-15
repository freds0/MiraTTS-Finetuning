# MiraTTS Finetuning

A modular and easy-to-use repository for finetuning the MiraTTS text-to-speech model on custom datasets.

## Overview

This repository provides a clean, organized structure for training and using the MiraTTS model. The original Jupyter notebook has been refactored into separate, reusable modules for better maintainability and ease of use.

## Features

- Modular architecture with separate components for model loading, data processing, training, and inference
- Command-line interface for training and testing
- Configurable parameters via config file or command-line arguments
- Support for custom datasets from HuggingFace
- Support for local datasets (LJSpeech)
- Easy model upload to HuggingFace Hub
- GPU memory monitoring
- Multiple requirements files for different environments (Colab, CPU, Development)

## Repository Structure

```
Mira-TTS-Finetuning/
├── src/
│   └── mira_tts/
│       ├── __init__.py           # Package initialization
│       ├── config.py              # Configuration settings
│       ├── model_loader.py        # Model loading utilities
│       ├── audio_codec.py         # Audio encoding/decoding
│       ├── data_processor.py      # Dataset processing
│       ├── trainer.py             # Training logic
│       ├── inference.py           # Inference utilities
│       └── ljspeech_loader.py     # LJSpeech dataset loader
├── train.py                       # Training script (HuggingFace datasets)
├── train_ljspeech.py              # Training script (LJSpeech local)
├── test_model.py                  # Testing/inference script
├── test_ljspeech_simple.py        # Test LJSpeech data loading
├── requirements.txt               # Main dependencies (GPU)
├── requirements-colab.txt         # For Google Colab
├── requirements-cpu.txt           # For CPU-only environments
├── requirements-dev.txt           # Development dependencies
├── requirements-minimal.txt       # Minimal for testing
├── setup.py                       # Package setup
├── README.md                      # This file
├── REQUIREMENTS.md                # Detailed requirements guide
├── INSTALL.md                     # Installation guide
├── LJSPEECH_TRAINING.md           # LJSpeech training guide
└── .gitignore                     # Git ignore file
```

## Installation

### Quick Start (Recommended for Beginners)

**Use Google Colab for easiest setup:**

```python
# In Google Colab
!git clone <repository-url>
%cd Mira-TTS-Finetuning
!pip install -r requirements-colab.txt
```

### Local Installation

#### 1. Clone the repository

```bash
git clone <repository-url>
cd Mira-TTS-Finetuning
```

#### 2. Choose the right requirements file

We provide multiple requirements files for different use cases:

- **`requirements.txt`** - Full installation for GPU training (recommended for local)
- **`requirements-colab.txt`** - For Google Colab
- **`requirements-cpu.txt`** - For CPU-only (no GPU)
- **`requirements-dev.txt`** - For development with extra tools
- **`requirements-minimal.txt`** - Minimal for testing data loading only

See [REQUIREMENTS.md](REQUIREMENTS.md) for detailed information.

#### 3. Install dependencies

For GPU training (local):
```bash
# Create conda environment (recommended)
conda create -n miratts python=3.10 -y
conda activate miratts

# Install PyTorch with CUDA
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Install other dependencies
pip install -r requirements.txt
```

Or install as a package:
```bash
pip install -e .
```

For detailed installation instructions and troubleshooting, see [INSTALL.md](INSTALL.md).

## Usage

### Training

#### Option 1: Training with HuggingFace Dataset

Basic training with default settings:
```bash
python train.py
```

Training with custom parameters:
```bash
python train.py \
    --dataset "your/dataset-name" \
    --num-samples 50 \
    --max-steps 100 \
    --learning-rate 1e-4 \
    --output-dir "my_model"
```

#### Option 2: Training with LJSpeech Local Dataset

```bash
python train_ljspeech.py \
    --ljspeech-path /path/to/LJSpeech-1.1 \
    --num-samples 50 \
    --max-steps 100 \
    --output-dir "outputs_ljspeech"
```

See [LJSPEECH_TRAINING.md](LJSPEECH_TRAINING.md) for detailed LJSpeech training guide.

#### Uploading to HuggingFace Hub:

```bash
python train.py \
    --push-to-hub \
    --hub-repo "username/model-name" \
    --hf-token "your_hf_token"
```

### Inference/Testing

#### Basic inference:

```bash
python test_model.py \
    --text "Hello, this is a test of the text to speech system." \
    --audio-file "path/to/reference_audio.wav" \
    --output "output.wav"
```

#### Inference with custom parameters:

```bash
python test_model.py \
    --text "Your text here" \
    --audio-file "reference.wav" \
    --output "result.wav" \
    --temperature 0.9 \
    --top-k 60 \
    --top-p 0.95 \
    --repetition-penalty 1.1
```

#### Using a custom trained model:

```bash
python test_model.py \
    --model-path "outputs/" \
    --text "Test with custom model" \
    --audio-file "reference.wav"
```

## Configuration

All configuration parameters can be modified in [src/mira_tts/config.py](src/mira_tts/config.py):

### Model Settings
- `MODEL_NAME`: HuggingFace model name
- `MAX_SEQ_LENGTH`: Maximum sequence length (30 seconds = 30 * 50)
- `DTYPE`: Data type (float32)

### Dataset Settings
- `DATASET_NAME`: HuggingFace dataset name
- `DATASET_SPLIT`: Dataset split to use
- `NUM_SAMPLES`: Number of samples to train on
- `SAMPLING_RATE`: Audio sampling rate (16000 Hz)

### Training Settings
- `PER_DEVICE_TRAIN_BATCH_SIZE`: Batch size per device
- `GRADIENT_ACCUMULATION_STEPS`: Gradient accumulation steps
- `MAX_STEPS`: Maximum training steps
- `LEARNING_RATE`: Learning rate
- `WEIGHT_DECAY`: Weight decay
- And more...

### Inference Settings
- `INFERENCE_TOP_K`: Top-k sampling
- `INFERENCE_TOP_P`: Nucleus sampling
- `INFERENCE_TEMPERATURE`: Sampling temperature
- `MAX_NEW_AUDIO_TOKENS`: Maximum audio tokens to generate

## Python API Usage

You can also use the modules programmatically:

```python
from src.mira_tts import Config, ModelLoader, DataProcessor, MiraTrainer, MiraInference

# Setup configuration
config = Config()
config.NUM_SAMPLES = 30
config.MAX_STEPS = 100

# Load model
model_loader = ModelLoader(config)
model, tokenizer = model_loader.load_model()

# Process data
data_processor = DataProcessor(config)
train_dataset = data_processor.process_dataset()

# Train
trainer = MiraTrainer(model, tokenizer, config)
trainer.setup_trainer(train_dataset)
trainer.train()
trainer.save_model()

# Inference
inference = MiraInference(model, tokenizer, config)
audio = inference.infer(
    text="Hello world",
    audio_file="reference.wav"
)
inference.save_audio(audio, "output.wav")
```

## Dataset Format

The dataset should be in HuggingFace datasets format with at least:
- `text`: Text transcription
- `audio`: Audio file

You can customize column names in the config:
```python
config.TEXT_COLUMN = "your_text_column"
config.AUDIO_COLUMN = "your_audio_column"
```

## Requirements

- Python >= 3.9, <= 3.11
- CUDA-capable GPU (recommended, 16GB+ VRAM for training)
- PyTorch >= 2.8.0
- Transformers == 4.56.2
- See [REQUIREMENTS.md](REQUIREMENTS.md) for detailed requirements guide
- See individual requirements-*.txt files for specific environments

## Notes

- The model works only in float32 or bfloat16 precision (NaNs in fp16)
- GPU with at least 16GB VRAM recommended for training
- For inference, GPUs with bfloat16 support (30xx series or above) are recommended
- Training on CPU is not recommended due to performance

## Troubleshooting

### Out of Memory
- Reduce `NUM_SAMPLES`
- Reduce `PER_DEVICE_TRAIN_BATCH_SIZE`
- Reduce `MAX_SEQ_LENGTH`

### Model NaNs
- Ensure you're using float32 or bfloat16 (not fp16)
- Check that `UNSLOTH_FORCE_FLOAT32` environment variable is set

### Audio Quality Issues
- Try adjusting inference parameters (temperature, top_k, top_p)
- Use higher quality reference audio
- Increase `MAX_NEW_AUDIO_TOKENS`

## License

Please refer to the original MiraTTS model license.

## Credits

- Original MiraTTS model by [YatharthS](https://huggingface.co/YatharthS)
- This repository refactors the training notebook into a modular structure

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
