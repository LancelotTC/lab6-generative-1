import os
from pathlib import Path

# Shared training/data constants
NUM_EPOCHS: int = 60
BATCH_SIZE: int = 64
IMAGE_SIZE: int = 64
TRAIN_VALID_RATIO: float = 0.95
MIN_INTENSITY_VALUE: float = 0.0
MAX_INTENSITY_VALUE: float = 1.0
NUM_WORKERS: int = 0 if os.name == "nt" else 4
SEED: int = 0
TSNE_RANDOM_STATE: int = 42

# Shared model constants used by both GAN and LDM autoencoder/discriminator parts.
SPATIAL_DIMS: int = 2
IN_CHANNELS: int = 1
OUT_CHANNELS: int = 1

AUTOENCODER_CHANNELS: tuple[int, ...] = (24, 48, 64)
LATENT_CHANNELS: int = 12
NUM_RES_BLOCKS: int = 3
ATTENTION_LEVELS: tuple[bool, ...] = (False, False, False)
WITH_ENCODER_NONLOCAL_ATTN: bool = False
WITH_DECODER_NONLOCAL_ATTN: bool = False

DISCRIMINATOR_NUM_LAYERS_D: int = 3
DISCRIMINATOR_CHANNELS: int = 16

AUTOENCODER_LEARNING_RATE: float = 1e-4
DISCRIMINATOR_LEARNING_RATE: float = 5e-4

KL_WEIGHT: float = 1e-5
PERCEPTUAL_WEIGHT: float = 1e-3
ADVERSARIAL_WEIGHT: float = 1e-2

INTERPOLATION_STEPS: int = 64
INTERMEDIATE_DECODE_DIVISOR: int = 20

DATA_ROOT_DIR: str = "data"
MEDNIST_DATA_DIR: str = "data/MedNIST"
MEDNIST_TRAIN_SECTION: str = "training"
MEDNIST_VALID_SECTION: str = "validation"
MEDNIST_LABELS: tuple[str, ...] = ("AbdomenCT", "BreastMRI", "ChestCT", "CXR", "Hand", "HeadCT")
SELECTED_LABEL: str = MEDNIST_LABELS[4]  # Hand
SELECTED_LABEL: str = MEDNIST_LABELS[3]  # CXR

# Run organization constants
RUNS_ROOT: Path = Path("runs")
MODEL_TYPES: tuple[str, ...] = ("GAN", "LDM")
RUN_DATETIME_FORMAT: str = "%Y%m%d_%H%M%S_%f"
