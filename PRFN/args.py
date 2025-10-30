import logging
import os
from dataclasses import dataclass
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class Config:
    # Data directories
    train_data_dir: Path = Path("D:/QB/Dataset/train/")
    eval_data_dir: Path = Path("D:/QB/Dataset/eval/")
    test_data_dir: Path = Path("D:/QB/Dataset/test/")

    # Output directories
    checkpoint_dir: Path = Path("./checkpoint")
    checkpoint_backup_dir: Path = Path("./checkpoint")
    record_dir: Path = Path("./log")
    log_dir: Path = Path("./log")
    output_dir: Path = Path("./result")

    # Model paths
    pretrained_model_path: Path = Path("./checkpoint/best.pth")

    # Training parameters
    max_value: int = 2047
    epochs: int = 300
    lr: float = 0.0005
    batch_size: int = 2
    lr_decay_freq: int = 100
    checkpoint_save_interval: int = 10
    validation_interval: int = 1
    seed: int = 42
    cuda: bool = True  # Use GPU if available
    gpu_ids: str = "0"  # Specify GPU IDs, e.g., "0", "0,1", or "1,2,3"

    def __post_init__(self):
        """Validate configuration, ensure directories exist, and set GPU configuration."""
        for field in self.__dataclass_fields__:
            if isinstance(getattr(self, field), str) and field not in ["gpu_ids"]:
                setattr(self, field, Path(getattr(self, field)))

        for dir_field in [
            "train_data_dir", "eval_data_dir", "test_data_dir",
            "checkpoint_dir", "checkpoint_backup_dir", "record_dir",
            "log_dir", "output_dir"
        ]:
            dir_path = getattr(self, dir_field)
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"Ensured directory exists: {dir_path}")
            except Exception as e:
                logger.warning(f"Failed to create directory {dir_path}: {e}")

        # Set GPU configuration
        if self.cuda:
            if self.gpu_ids:
                os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_ids
                logger.info(f"Set CUDA_VISIBLE_DEVICES to {self.gpu_ids}")
            else:
                logger.info("No specific GPU IDs provided, using all available GPUs")
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Disable GPU
            logger.info("CUDA disabled, using CPU")

# Instantiate the configuration
config = Config()