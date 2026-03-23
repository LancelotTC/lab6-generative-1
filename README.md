# MLCV Lab 6 - Generative Models on MedNIST

## 1. Scope

This repository contains two training pipelines for class-conditional MedNIST image generation (single selected class):

- `GAN.py`: adversarially trained `AutoencoderKL` with `PatchDiscriminator`
- `LDM.py`: adversarially trained `AutoencoderKL` (stage 1) followed by latent diffusion UNet training (stage 2)

Both pipelines share common data handling and shared architecture/training constants, and each run is recorded under a timestamped directory.

## 2. Repository Structure

```text
.
+-- GAN.py
+-- LDM.py
+-- common.py
+-- settings.py
+-- utils.py
+-- model_structures.py
`-- runs/
    +-- GAN/
    `-- LDM/
```

### File roles

- `settings.py`: shared constants (data, common model hyperparameters, run layout)
- `common.py`: shared utilities (MedNIST loaders, plotting helpers, latent-space projection, perceptual loss setup)
- `utils.py`: generic utilities (run directory creation, JSON save, GIF save, numeric helper)
- `GAN.py`: GAN/VAE-GAN training and post-training visualizations
- `LDM.py`: LDM training (AE stage + diffusion stage) and post-training visualizations
- `model_structures.py`: utility script to dump architecture summaries

## 3. Environment

- Python: `3.13.x` (validated on `3.13.2`)
- Main libraries:
  - `torch`
  - `torchvision`
  - `monai`
  - `numpy`
  - `matplotlib`
  - `scikit-learn`
  - `tqdm`
  - `torchinfo`
  - `pillow`

Install dependencies:

```bash
python -m pip install torch torchvision monai numpy matplotlib scikit-learn tqdm torchinfo pillow
```

## 4. Data

Data loading is implemented in `common.py` via `MedNISTDataset`.

- dataset root: `data/`
- selected class: `settings.py -> SELECTED_LABEL`
- preprocessing:
  - channel-first conversion
  - intensity scaling to `[0, 1]`
  - resize to `IMAGE_SIZE x IMAGE_SIZE`
  - training-only affine augmentation (`RandAffined`)

## 5. Configuration

Global/shared settings are defined in `settings.py`, including:

- training/data constants (`NUM_EPOCHS`, `BATCH_SIZE`, `IMAGE_SIZE`, `SEED`)
- selected MedNIST class (`SELECTED_LABEL`, default: `Hand`)
- shared model constants (`AUTOENCODER_CHANNELS`, `LATENT_CHANNELS`, `NUM_RES_BLOCKS`, discriminator constants)
- shared loss weights (`KL_WEIGHT`, `PERCEPTUAL_WEIGHT`, `ADVERSARIAL_WEIGHT`)
- run directory format (`RUNS_ROOT`, `MODEL_TYPES`, `RUN_DATETIME_FORMAT`)

Model-specific constants remain in:

- `GAN.py` for GAN-specific behavior
- `LDM.py` for diffusion-specific behavior (UNet/scheduler/epoch settings)

## 6. Training Execution

Run GAN:

```bash
python GAN.py
```

Run LDM:

```bash
python LDM.py
```

## 7. Run Logging and Artifacts

Each execution creates a unique directory:

`runs/<MODEL_TYPE>/<YYYYMMDD_HHMMSS_microseconds>/`

Per-run structure:

```text
<run_dir>/
+-- hyperparameters.json
+-- metrics.json
+-- plots/
`-- models/
```

`hyperparameters.json` includes `common.selected_label`, which stores the MedNIST class used for that run.

### GAN artifacts

`plots/`:

- `GAN - Training Metrics.png`
- `GAN - Adversarial Training Curves.png`
- `GAN - Latent Space (t-SNE 2D).png`
- `GAN - Latent Interpolation.gif`
- `GAN - Deterministic Reference Reconstructions.png`
- `GAN - Deterministic Reference Reconstructions.json`

`models/`:

- `best_test_lossmodel.pth`

### LDM artifacts

`plots/`:

- `Diffusion Model - Autoencoder Training Metrics.png`
- `Diffusion Model - Autoencoder Train vs Validation Reconstruction.png`
- `Diffusion Model - Adversarial Training Curves.png`
- `Diffusion Model - Diffusion Training Metrics.png`
- `Diffusion Model - Diffusion Train vs Validation.png`
- `Diffusion Model - Latent Space (t-SNE 2D).png`
- `Diffusion Model - Decoded Intermediates Every 100 Steps.png`
- `MedNIST Interpolation.gif`
- `LDM - Deterministic Reference Reconstructions.png`
- `LDM - Deterministic Reference Reconstructions.json`

`models/`:

- `autoencoderkl_for_diffusion_state_dict.pth`
- `diffusion_model_unet_state_dict.pth`
- `generated_sample.pt`

## 8. Deterministic Reconstruction Utility

Generate deterministic 10-image reference-vs-reconstruction comparisons for all loadable runs:

```bash
python reconstruct_reference_samples.py
```

Behavior:

- scans every run under `runs/GAN/*` and `runs/LDM/*`
- loads each run checkpoint and reconstructs 10 deterministic validation images
- uses run-specific label metadata from `hyperparameters.json -> common.selected_label` (or `settings.SELECTED_LABEL` when metadata is absent)
- saves outputs under each run's `plots/`

## 9. Notes on Method

- GAN pipeline: adversarial VAE-style autoencoder training with reconstruction, KL, perceptual, and adversarial losses.
- LDM pipeline:
  - stage 1: adversarial autoencoder training
  - stage 2: diffusion UNet training in latent space with DDPM scheduler
- Latent-space visualization uses t-SNE projection with equal x/y axis scaling for geometric consistency in 2D plots.

## 10. Additions Relative to the Original Baseline Code

Compared with the provided baseline implementation, the project includes:

- Separation of concerns into reusable modules:
  - shared constants in `settings.py`
  - shared data/plotting/perceptual-loss/latent-space utilities in `common.py`
  - generic run-management and serialization helpers in `utils.py`
- Class-based training organization in both pipelines (`Components`, `Training`, `Visualization`) with explicit stage boundaries.
- Consistent run tracking by model type and timestamped run directory:
  - `runs/GAN/<datetime>/`
  - `runs/LDM/<datetime>/`
- Structured artifact storage per run:
  - `plots/` for visual outputs
  - `models/` for `.pt/.pth` model artifacts
  - run-root JSON metadata (`hyperparameters.json`, `metrics.json`)
- Persistent experiment logging:
  - saved hyperparameter snapshots per run
  - saved training/validation metrics and run duration
- Standardized visualization outputs for reporting and comparison:
  - adversarial training curves
  - latent-space 2D projections (t-SNE)
  - interpolation outputs
  - diffusion intermediate decoding strips
- LDM integration into the same workflow and output conventions as GAN (shared run layout, logging, and plotting interfaces).
- Mixed-precision training support integrated in LDM training stages.
- Full type annotations across shared and pipeline functions to support maintainability and development.
