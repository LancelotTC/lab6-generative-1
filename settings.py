import os
from pathlib import Path

# Shared training/data constants
COMMON_NUM_EPOCHS: int = 60
BATCH_SIZE: int = 32
IMAGE_SIZE: int = 64
TRAIN_VALID_RATIO: float = 0.95
MIN_INTENSITY_VALUE: float = 0.0
MAX_INTENSITY_VALUE: float = 1.0
NUM_WORKERS: int = 0 if os.name == "nt" else 4
SEED: int = 0
COMMON_TSNE_RANDOM_STATE: int = 42

DATA_ROOT_DIR: str = "data"
MEDNIST_DATA_DIR: str = "data/MedNIST"
MEDNIST_TRAIN_SECTION: str = "training"
MEDNIST_VALID_SECTION: str = "validation"
MEDNIST_LABELS: tuple[str, ...] = ("AbdomenCT", "BreastMRI", "ChestCT", "CXR", "Hand", "HeadCT")
SELECTED_LABEL: str = MEDNIST_LABELS[4]  # Hand

# Run organization constants
RUNS_ROOT: Path = Path("runs")
MODEL_TYPES: tuple[str, ...] = ("GAN", "LDM")
RUN_DATETIME_FORMAT: str = "%Y%m%d_%H%M%S_%f"
