"""
Setup script for MiraTTS Finetuning
"""
from setuptools import setup, find_packages

setup(
    name="mira-tts-finetuning",
    version="0.1.0",
    description="Finetuning pipeline for MiraTTS text-to-speech model",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.8.0",
        "transformers==4.56.2",
        "accelerate==1.8.1",
        "datasets>=3.4.1,<4.0.0",
        "unsloth",
        "trl",
        "bitsandbytes",
        "peft",
        "sentencepiece",
        "protobuf",
        "huggingface_hub>=0.34.0",
        "hf_transfer",
        "omegaconf",
        "einx",
        "torchcodec",
        "librosa",
        "numpy",
        "onnxruntime-gpu",
    ],
    extras_require={
        "dev": [
            "pytest",
            "black",
            "flake8",
            "mypy",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
