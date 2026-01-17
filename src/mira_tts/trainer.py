"""
Training module for MiraTTS
"""
import os
import torch
from trl import SFTTrainer, SFTConfig
from .config import Config


class MiraTrainer:
    """Class for training MiraTTS model"""

    def __init__(self, model, tokenizer, config: Config = None):
        """
        Initialize MiraTrainer

        Args:
            model: The model to train
            tokenizer: The tokenizer
            config: Configuration object. If None, uses default Config.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or Config()
        self.trainer = None
        self._setup_logging()

    def _setup_logging(self):
        """Setup logging integrations (wandb and tensorboard)"""
        report_to = []

        # Setup WandB
        if self.config.USE_WANDB:
            try:
                import wandb

                # Set API key if provided
                if self.config.WANDB_API_KEY:
                    os.environ["WANDB_API_KEY"] = self.config.WANDB_API_KEY

                # Initialize wandb
                run_name = self.config.WANDB_RUN_NAME or f"miratts-{self.config.OUTPUT_DIR.split('/')[-1]}"

                wandb.init(
                    project=self.config.WANDB_PROJECT,
                    name=run_name,
                    entity=self.config.WANDB_ENTITY,
                    config={
                        "learning_rate": self.config.LEARNING_RATE,
                        "batch_size": self.config.PER_DEVICE_TRAIN_BATCH_SIZE,
                        "max_steps": self.config.MAX_STEPS,
                        "model": self.config.MODEL_NAME,
                        "num_samples": self.config.NUM_SAMPLES,
                    }
                )

                report_to.append("wandb")
                print(f"✓ WandB initialized: {self.config.WANDB_PROJECT}/{run_name}")

            except ImportError:
                print("⚠ WandB requested but not installed. Run: pip install wandb")
                print("  Continuing without WandB...")
            except Exception as e:
                print(f"⚠ Failed to initialize WandB: {e}")
                print("  Continuing without WandB...")

        # Setup TensorBoard
        if self.config.USE_TENSORBOARD:
            try:
                from torch.utils.tensorboard import SummaryWriter

                # Create tensorboard directory
                tb_dir = os.path.join(self.config.TENSORBOARD_LOG_DIR, self.config.OUTPUT_DIR.split('/')[-1])
                os.makedirs(tb_dir, exist_ok=True)

                report_to.append("tensorboard")
                print(f"✓ TensorBoard enabled: {tb_dir}")
                print(f"  View with: tensorboard --logdir={self.config.TENSORBOARD_LOG_DIR}")

            except ImportError:
                print("⚠ TensorBoard requested but not installed. Run: pip install tensorboard")
                print("  Continuing without TensorBoard...")

        # Update config
        if report_to:
            self.config.REPORT_TO = report_to
        elif self.config.REPORT_TO == "none":
            self.config.REPORT_TO = "none"

        print(f"Logging to: {self.config.REPORT_TO}")

    def setup_trainer(self, train_dataset):
        """
        Setup the SFT trainer

        Args:
            train_dataset: Dataset for training

        Returns:
            SFTTrainer: Configured trainer
        """
        print("Setting up trainer...")

        sft_config = SFTConfig(
            per_device_train_batch_size=self.config.PER_DEVICE_TRAIN_BATCH_SIZE,
            gradient_accumulation_steps=self.config.GRADIENT_ACCUMULATION_STEPS,
            warmup_steps=self.config.WARMUP_STEPS,
            max_steps=self.config.MAX_STEPS,
            learning_rate=self.config.LEARNING_RATE,
            fp16=self.config.FP16,
            bf16=self.config.BF16,
            logging_steps=self.config.LOGGING_STEPS,
            optim=self.config.OPTIM,
            weight_decay=self.config.WEIGHT_DECAY,
            lr_scheduler_type=self.config.LR_SCHEDULER_TYPE,
            seed=self.config.SEED,
            output_dir=self.config.OUTPUT_DIR,
            report_to=self.config.REPORT_TO,
        )

        self.trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=train_dataset,
            dataset_text_field="text",
            max_seq_length=self.config.MAX_SEQ_LENGTH,
            packing=self.config.PACKING,
            args=sft_config,
        )

        print("Trainer setup complete!")
        return self.trainer

    def train(self):
        """
        Start training the model

        Returns:
            dict: Training statistics
        """
        if self.trainer is None:
            raise ValueError("Trainer not initialized. Call setup_trainer first.")

        print("Starting training...")
        trainer_stats = self.trainer.train()
        print("Training complete!")

        # Log final metrics
        self._log_final_metrics(trainer_stats)

        return trainer_stats

    def _log_final_metrics(self, trainer_stats):
        """Log final training metrics to wandb"""
        if self.config.USE_WANDB:
            try:
                import wandb

                # Log final metrics
                wandb.log({
                    "final/train_loss": trainer_stats.training_loss,
                    "final/train_runtime": trainer_stats.metrics.get("train_runtime", 0),
                    "final/train_samples_per_second": trainer_stats.metrics.get("train_samples_per_second", 0),
                })

                print("✓ Final metrics logged to WandB")
            except Exception as e:
                print(f"⚠ Failed to log final metrics: {e}")

    def print_gpu_stats(self):
        """Print GPU memory statistics"""
        if not torch.cuda.is_available():
            print("CUDA not available")
            return

        gpu_stats = torch.cuda.get_device_properties(0)
        start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)

        print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
        print(f"{start_gpu_memory} GB of memory reserved.")

    def save_model(self, output_path: str = None):
        """
        Save the trained model

        Args:
            output_path: Path to save the model. If None, uses config OUTPUT_DIR.
        """
        output_path = output_path or self.config.OUTPUT_DIR
        print(f"Saving model to {output_path}...")
        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        print("Model saved!")

    def push_to_hub(self, repo_name: str = None, token: str = None):
        """
        Push model to HuggingFace Hub

        Args:
            repo_name: Repository name. If None, uses config value.
            token: HuggingFace token. If None, uses config value.
        """
        from huggingface_hub import login

        repo_name = repo_name or self.config.UPLOAD_MODEL_REPO
        token = token or self.config.HF_TOKEN

        if not token:
            raise ValueError("HuggingFace token required. Set HF_TOKEN in config or pass as argument.")

        print(f"Logging in to HuggingFace...")
        login(token)

        print(f"Pushing model to {repo_name}...")
        self.model.push_to_hub(repo_name)
        self.tokenizer.push_to_hub(repo_name)
        print("Model pushed to hub successfully!")
