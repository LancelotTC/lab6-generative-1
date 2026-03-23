from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from monai.apps import MedNISTDataset
from monai.data import Dataset
from monai.networks.nets import AutoencoderKL
from monai.transforms import Compose, EnsureChannelFirstd, LoadImaged, Resized, ScaleIntensityRanged

from settings import (
    DATA_ROOT_DIR,
    IMAGE_SIZE,
    MAX_INTENSITY_VALUE,
    MEDNIST_DATA_DIR,
    MEDNIST_VALID_SECTION,
    MIN_INTENSITY_VALUE,
    SEED,
    SELECTED_LABEL,
)

SAMPLE_COUNT = 10
RUNS_ROOT = Path("runs")


@dataclass(frozen=True)
class ReconstructionRun:
    model_type: str
    run_dir: Path
    checkpoint_path: Path
    score: float


def _load_json(file_path: Path) -> dict[str, Any]:
    return json.loads(file_path.read_text(encoding="utf-8"))


def _find_best_loadable_run(model_type: str) -> ReconstructionRun | None:
    model_root = RUNS_ROOT / model_type
    if not model_root.exists():
        return None

    if model_type == "GAN":
        checkpoint_name = "best_test_lossmodel.pth"

        def get_score(metrics: dict[str, Any]) -> float:
            return float(metrics["best_valid_metric"])

    elif model_type == "LDM":
        checkpoint_name = "autoencoderkl_for_diffusion_state_dict.pth"

        def get_score(metrics: dict[str, Any]) -> float:
            val_losses = metrics.get("autoencoder", {}).get("val_recon_losses", [])
            return min(float(loss) for loss in val_losses) if val_losses else math.inf

    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    best_run: ReconstructionRun | None = None

    for run_dir in sorted(model_root.glob("*")):
        if not run_dir.is_dir():
            continue

        metrics_path = run_dir / "metrics.json"
        checkpoint_path = run_dir / "models" / checkpoint_name

        if not metrics_path.exists() or not checkpoint_path.exists():
            continue

        score = get_score(_load_json(metrics_path))
        if not math.isfinite(score):
            continue

        candidate = ReconstructionRun(
            model_type=model_type,
            run_dir=run_dir,
            checkpoint_path=checkpoint_path,
            score=score,
        )
        if best_run is None or candidate.score < best_run.score:
            best_run = candidate

    return best_run


def _build_autoencoder_from_hyperparameters(
    model_type: str,
    hyperparameters: dict[str, Any],
    device: torch.device,
) -> AutoencoderKL:
    if model_type == "GAN":
        config = hyperparameters["gan_specific"]
        channels = tuple(config["channels"])
    elif model_type == "LDM":
        config = hyperparameters["ldm_specific"]
        channels = tuple(config["autoencoder_channels"])
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    model = AutoencoderKL(
        spatial_dims=int(config.get("spatial_dims", 2)),
        in_channels=int(config.get("in_channels", 1)),
        out_channels=int(config.get("out_channels", 1)),
        channels=channels,
        latent_channels=int(config["latent_channels"]),
        num_res_blocks=config["num_res_blocks"],
        norm_num_groups=int(config["norm_num_groups"]),
        attention_levels=tuple(bool(level) for level in config["attention_levels"]),
        with_encoder_nonlocal_attn=bool(config.get("with_encoder_nonlocal_attn", False)),
        with_decoder_nonlocal_attn=bool(config.get("with_decoder_nonlocal_attn", False)),
    )
    model.to(device)
    model.eval()
    return model


def _select_reference_images(sample_count: int, seed: int) -> tuple[torch.Tensor, list[int]]:
    download = not os.path.exists(MEDNIST_DATA_DIR)
    source_dataset = MedNISTDataset(
        root_dir=DATA_ROOT_DIR,
        section=MEDNIST_VALID_SECTION,
        download=download,
        seed=SEED,
        cache_rate=0.0,
        num_workers=0,
        progress=False,
    )
    datalist = [
        {"image": item["image"], "label": SELECTED_LABEL}
        for item in source_dataset.data
        if item["class_name"] == SELECTED_LABEL
    ]

    eval_transforms = Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=0.0,
                a_max=255.0,
                b_min=MIN_INTENSITY_VALUE,
                b_max=MAX_INTENSITY_VALUE,
                clip=True,
            ),
            Resized(keys=["image"], spatial_size=[IMAGE_SIZE, IMAGE_SIZE]),
        ]
    )
    dataset = Dataset(data=datalist, transform=eval_transforms)

    if len(dataset) == 0:
        raise ValueError("The selected test dataset is empty.")

    count = min(sample_count, len(dataset))
    generator = torch.Generator().manual_seed(seed)
    selected_indices = torch.randperm(len(dataset), generator=generator)[:count].tolist()

    images = torch.stack([dataset[index]["image"] for index in selected_indices], dim=0)
    return images, selected_indices


def _reconstruct_with_mean_latent(model: AutoencoderKL, images: torch.Tensor, device: torch.device) -> torch.Tensor:
    with torch.no_grad():
        latent_mu, _ = model.encode(images.to(device))
        reconstructions = model.decode(latent_mu)
    return reconstructions.detach().cpu().clamp_(0.0, 1.0)


def _save_reconstruction_panel(
    model_type: str,
    run_info: ReconstructionRun,
    originals: torch.Tensor,
    reconstructions: torch.Tensor,
    sample_indices: list[int],
) -> None:
    plots_dir = run_info.run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    originals_np = originals.squeeze(1).numpy()
    reconstructions_np = reconstructions.squeeze(1).numpy()
    errors_np = np.abs(originals_np - reconstructions_np)
    per_image_mae = errors_np.mean(axis=(1, 2))

    figure, axes = plt.subplots(3, len(sample_indices), figsize=(1.9 * len(sample_indices), 5.8))
    for column, dataset_index in enumerate(sample_indices):
        axes[0, column].imshow(originals_np[column], cmap="gray", vmin=0.0, vmax=1.0)
        axes[1, column].imshow(reconstructions_np[column], cmap="gray", vmin=0.0, vmax=1.0)
        axes[2, column].imshow(errors_np[column], cmap="magma", vmin=0.0, vmax=max(0.1, float(errors_np.max())))

        axes[0, column].set_title(f"idx {dataset_index}")
        axes[1, column].set_title(f"MAE {per_image_mae[column]:.4f}")

        for row in range(3):
            axes[row, column].axis("off")

    axes[0, 0].set_ylabel("Original", fontsize=11)
    axes[1, 0].set_ylabel("Reconstruction", fontsize=11)
    axes[2, 0].set_ylabel("Abs Error", fontsize=11)

    figure.suptitle(
        f"{model_type} deterministic reconstructions\n"
        f"run={run_info.run_dir.name} | score={run_info.score:.6f} | label={SELECTED_LABEL} | mean MAE={per_image_mae.mean():.4f}",
        fontsize=13,
    )
    figure.tight_layout()

    output_path = plots_dir / f"{model_type} - Deterministic Reference Reconstructions.png"
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)

    summary = {
        "model_type": model_type,
        "run_dir": str(run_info.run_dir),
        "checkpoint_path": str(run_info.checkpoint_path),
        "selection_seed": SEED,
        "selected_label": SELECTED_LABEL,
        "sample_indices": sample_indices,
        "selection_dataset": "MedNIST validation section with deterministic no-cache preprocessing",
        "score": run_info.score,
        "mean_mae": float(per_image_mae.mean()),
        "per_image_mae": [float(value) for value in per_image_mae],
    }
    summary_path = plots_dir / f"{model_type} - Deterministic Reference Reconstructions.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reference_images, sample_indices = _select_reference_images(sample_count=SAMPLE_COUNT, seed=SEED)
    print(f"Selected {len(sample_indices)} deterministic reference images for label '{SELECTED_LABEL}': {sample_indices}")

    for model_type in ("GAN", "LDM"):
        run_info = _find_best_loadable_run(model_type)
        if run_info is None:
            print(f"[skip] {model_type}: no run with both metrics and a reconstruction checkpoint was found.")
            continue

        hyperparameters = _load_json(run_info.run_dir / "hyperparameters.json")
        model = _build_autoencoder_from_hyperparameters(model_type, hyperparameters, device)
        state_dict = torch.load(run_info.checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)

        reconstructions = _reconstruct_with_mean_latent(model, reference_images, device)
        _save_reconstruction_panel(
            model_type=model_type,
            run_info=run_info,
            originals=reference_images,
            reconstructions=reconstructions,
            sample_indices=sample_indices,
        )
        print(
            f"[saved] {model_type}: {run_info.run_dir / 'plots' / f'{model_type} - Deterministic Reference Reconstructions.png'}"
        )


if __name__ == "__main__":
    main()
