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
from monai.data import DataLoader, Dataset
from monai.inferers import LatentDiffusionInferer
from monai.networks.nets import AutoencoderKL, DiffusionModelUNet
from monai.networks.schedulers import DDPMScheduler as LDMScheduler
from monai.transforms import Compose, EnsureChannelFirstd, LoadImaged, RandAffined, Resized, ScaleIntensityRanged
from torch.utils.data import random_split

from common import (
    collect_latent_vectors,
    save_latent_space_plot,
    save_metric_panels,
    save_two_curve_plot,
)
from settings import (
    BATCH_SIZE,
    DATA_ROOT_DIR,
    IMAGE_SIZE,
    INTERMEDIATE_DECODE_DIVISOR,
    INTERPOLATION_STEPS,
    MAX_INTENSITY_VALUE,
    MEDNIST_DATA_DIR,
    MEDNIST_VALID_SECTION,
    MIN_INTENSITY_VALUE,
    RUNS_ROOT,
    SEED,
    SELECTED_LABEL,
    TRAIN_VALID_RATIO,
)
from utils import save_animation_as_gif

SAMPLE_COUNT = 10


@dataclass(frozen=True)
class RunInfo:
    model_type: str
    run_dir: Path
    hyperparameters: dict[str, Any]
    metrics: dict[str, Any]
    auto_checkpoint_path: Path
    diffusion_checkpoint_path: Path | None


_DATALOADER_CACHE: dict[tuple[str, int, int, float], tuple[DataLoader, DataLoader, DataLoader]] = {}
_REFERENCE_CACHE: dict[tuple[str, int], tuple[torch.Tensor, list[int]]] = {}


def _load_json(file_path: Path) -> dict[str, Any]:
    return json.loads(file_path.read_text(encoding="utf-8"))


def _pgcd(*values: int) -> int:
    result = 0
    for value in values:
        result = math.gcd(result, int(value))
    return result


def _collect_runs(model_type: str) -> list[RunInfo]:
    model_root = RUNS_ROOT / model_type
    if not model_root.exists():
        return []

    if model_type == "GAN":
        auto_checkpoint_name = "best_test_lossmodel.pth"
        diffusion_checkpoint_name = None
    elif model_type == "LDM":
        auto_checkpoint_name = "autoencoderkl_for_diffusion_state_dict.pth"
        diffusion_checkpoint_name = "diffusion_model_unet_state_dict.pth"
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    runs: list[RunInfo] = []
    for run_dir in sorted([path for path in model_root.iterdir() if path.is_dir()], key=lambda path: path.name):
        hyperparameters_path = run_dir / "hyperparameters.json"
        metrics_path = run_dir / "metrics.json"
        auto_checkpoint_path = run_dir / "models" / auto_checkpoint_name

        if not hyperparameters_path.exists() or not metrics_path.exists() or not auto_checkpoint_path.exists():
            continue

        diffusion_checkpoint_path = None
        if diffusion_checkpoint_name is not None:
            candidate = run_dir / "models" / diffusion_checkpoint_name
            if candidate.exists():
                diffusion_checkpoint_path = candidate

        runs.append(
            RunInfo(
                model_type=model_type,
                run_dir=run_dir,
                hyperparameters=_load_json(hyperparameters_path),
                metrics=_load_json(metrics_path),
                auto_checkpoint_path=auto_checkpoint_path,
                diffusion_checkpoint_path=diffusion_checkpoint_path,
            )
        )

    return runs


def _resolve_run_common(hyperparameters: dict[str, Any]) -> tuple[str, int, int, float]:
    common = hyperparameters.get("common", {})
    selected_label = str(common.get("selected_label", SELECTED_LABEL))
    batch_size = int(common.get("batch_size", BATCH_SIZE))
    image_size = int(common.get("image_size", IMAGE_SIZE))
    train_valid_ratio = float(common.get("train_valid_ratio", TRAIN_VALID_RATIO))
    return selected_label, batch_size, image_size, train_valid_ratio


def _get_cached_dataloaders(
    selected_label: str,
    batch_size: int,
    image_size: int,
    train_valid_ratio: float,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    cache_key = (selected_label, batch_size, image_size, train_valid_ratio)
    if cache_key not in _DATALOADER_CACHE:
        _DATALOADER_CACHE[cache_key] = _build_mednist_dataloaders(
            batch_size=batch_size,
            image_size=image_size,
            train_valid_ratio=train_valid_ratio,
            selected_label=selected_label,
        )
    return _DATALOADER_CACHE[cache_key]


def _build_mednist_dataloaders(
    batch_size: int,
    image_size: int,
    train_valid_ratio: float,
    selected_label: str,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    download = not os.path.exists(MEDNIST_DATA_DIR)
    train_set = MedNISTDataset(
        root_dir=DATA_ROOT_DIR,
        section="training",
        download=download,
        seed=SEED,
        cache_rate=0.0,
        num_workers=0,
        progress=False,
    )
    test_set = MedNISTDataset(
        root_dir=DATA_ROOT_DIR,
        section=MEDNIST_VALID_SECTION,
        download=download,
        seed=SEED,
        cache_rate=0.0,
        num_workers=0,
        progress=False,
    )

    train_datalist = [
        {"image": item["image"], "label": selected_label}
        for item in train_set.data
        if item["class_name"] == selected_label
    ]
    test_datalist = [
        {"image": item["image"], "label": selected_label}
        for item in test_set.data
        if item["class_name"] == selected_label
    ]

    base_transforms = [
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
        Resized(keys=["image"], spatial_size=[image_size, image_size]),
    ]
    train_transforms = Compose(
        base_transforms
        + [
            RandAffined(
                keys=["image"],
                rotate_range=[(-np.pi / 36, np.pi / 36), (-np.pi / 36, np.pi / 36)],
                translate_range=[(-1, 1), (-1, 1)],
                scale_range=[(-0.05, 0.05), (-0.05, 0.05)],
                spatial_size=[image_size, image_size],
                padding_mode="zeros",
                prob=0.5,
            )
        ]
    )
    eval_transforms = Compose(base_transforms)

    train_size = int(train_valid_ratio * len(train_datalist))
    valid_size = len(train_datalist) - train_size
    split_generator = torch.Generator().manual_seed(SEED)
    train_subset, valid_subset = random_split(train_datalist, [train_size, valid_size], generator=split_generator)

    train_data = [train_datalist[index] for index in train_subset.indices]
    valid_data = [train_datalist[index] for index in valid_subset.indices]

    train_dataset = Dataset(data=train_data, transform=train_transforms)
    valid_dataset = Dataset(data=valid_data, transform=eval_transforms)
    test_dataset = Dataset(data=test_datalist, transform=eval_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, persistent_workers=False)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0, persistent_workers=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, persistent_workers=False)

    return train_loader, valid_loader, test_loader


def _select_reference_images(sample_count: int, seed: int, selected_label: str, image_size: int) -> tuple[torch.Tensor, list[int]]:
    download = not os.path.exists(MEDNIST_DATA_DIR)
    source_dataset = MedNISTDataset(
        root_dir=DATA_ROOT_DIR,
        section=MEDNIST_VALID_SECTION,
        download=download,
        seed=seed,
        cache_rate=0.0,
        num_workers=0,
        progress=False,
    )
    datalist = [
        {"image": item["image"], "label": selected_label}
        for item in source_dataset.data
        if item["class_name"] == selected_label
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
            Resized(keys=["image"], spatial_size=[image_size, image_size]),
        ]
    )
    dataset = Dataset(data=datalist, transform=eval_transforms)

    if len(dataset) == 0:
        raise ValueError(f"The selected validation dataset is empty for label '{selected_label}'.")

    count = min(sample_count, len(dataset))
    generator = torch.Generator().manual_seed(seed)
    selected_indices = torch.randperm(len(dataset), generator=generator)[:count].tolist()

    images = torch.stack([dataset[index]["image"] for index in selected_indices], dim=0)
    return images, selected_indices


def _get_cached_reference_images(selected_label: str, image_size: int) -> tuple[torch.Tensor, list[int]]:
    cache_key = (selected_label, image_size)
    if cache_key not in _REFERENCE_CACHE:
        _REFERENCE_CACHE[cache_key] = _select_reference_images(
            sample_count=SAMPLE_COUNT,
            seed=SEED,
            selected_label=selected_label,
            image_size=image_size,
        )
    return _REFERENCE_CACHE[cache_key]


def _build_autoencoder_from_hyperparameters(
    model_type: str,
    hyperparameters: dict[str, Any],
    device: torch.device,
    with_encoder_nonlocal_attn_override: bool | None = None,
    with_decoder_nonlocal_attn_override: bool | None = None,
) -> AutoencoderKL:
    if model_type == "GAN":
        config = hyperparameters["gan_specific"]
        channels = tuple(config["channels"])
    elif model_type == "LDM":
        config = hyperparameters["ldm_specific"]
        channels = tuple(config["autoencoder_channels"])
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    with_encoder_nonlocal_attn = bool(config.get("with_encoder_nonlocal_attn", False))
    with_decoder_nonlocal_attn = bool(config.get("with_decoder_nonlocal_attn", False))
    if with_encoder_nonlocal_attn_override is not None:
        with_encoder_nonlocal_attn = with_encoder_nonlocal_attn_override
    if with_decoder_nonlocal_attn_override is not None:
        with_decoder_nonlocal_attn = with_decoder_nonlocal_attn_override

    model = AutoencoderKL(
        spatial_dims=int(config.get("spatial_dims", 2)),
        in_channels=int(config.get("in_channels", 1)),
        out_channels=int(config.get("out_channels", 1)),
        channels=channels,
        latent_channels=int(config["latent_channels"]),
        num_res_blocks=int(config["num_res_blocks"]),
        norm_num_groups=int(config["norm_num_groups"]),
        attention_levels=tuple(bool(level) for level in config["attention_levels"]),
        with_encoder_nonlocal_attn=with_encoder_nonlocal_attn,
        with_decoder_nonlocal_attn=with_decoder_nonlocal_attn,
    )
    model.to(device)
    model.eval()
    return model


def _load_autoencoder_for_run(run_info: RunInfo, device: torch.device) -> AutoencoderKL:
    state_dict = torch.load(run_info.auto_checkpoint_path, map_location=device)
    model = _build_autoencoder_from_hyperparameters(
        model_type=run_info.model_type,
        hyperparameters=run_info.hyperparameters,
        device=device,
    )

    try:
        model.load_state_dict(state_dict, strict=True)
        return model
    except RuntimeError as first_error:
        encoder_has_attn = any(key.startswith("encoder") and ".attn." in key for key in state_dict)
        decoder_has_attn = any(key.startswith("decoder") and ".attn." in key for key in state_dict)

        config_key = "gan_specific" if run_info.model_type == "GAN" else "ldm_specific"
        config = run_info.hyperparameters.get(config_key, {})
        current_encoder_nonlocal = bool(config.get("with_encoder_nonlocal_attn", False))
        current_decoder_nonlocal = bool(config.get("with_decoder_nonlocal_attn", False))

        should_retry = (
            encoder_has_attn != current_encoder_nonlocal
            or decoder_has_attn != current_decoder_nonlocal
        )
        if not should_retry:
            raise first_error

        fallback_model = _build_autoencoder_from_hyperparameters(
            model_type=run_info.model_type,
            hyperparameters=run_info.hyperparameters,
            device=device,
            with_encoder_nonlocal_attn_override=encoder_has_attn,
            with_decoder_nonlocal_attn_override=decoder_has_attn,
        )
        fallback_model.load_state_dict(state_dict, strict=True)
        return fallback_model


def _build_ldm_unet_from_hyperparameters(hyperparameters: dict[str, Any], device: torch.device) -> DiffusionModelUNet:
    config = hyperparameters["ldm_specific"]
    channels = tuple(int(channel) for channel in config["diffusion_channels"])
    norm_num_groups = int(config.get("diffusion_norm_num_groups", _pgcd(*channels)))

    unet = DiffusionModelUNet(
        spatial_dims=int(config.get("spatial_dims", 2)),
        in_channels=int(config["latent_channels"]),
        out_channels=int(config["latent_channels"]),
        num_res_blocks=int(config["num_res_blocks"]),
        channels=channels,
        attention_levels=tuple(bool(level) for level in config["diffusion_attention_levels"]),
        num_head_channels=tuple(int(value) for value in config["diffusion_num_head_channels"]),
        norm_num_groups=norm_num_groups,
    )
    unet.to(device)
    unet.eval()
    return unet


def _build_scheduler_from_hyperparameters(hyperparameters: dict[str, Any]) -> LDMScheduler:
    config = hyperparameters["ldm_specific"]
    return LDMScheduler(
        num_train_timesteps=int(config["diffusion_num_train_timesteps"]),
        schedule=str(config["diffusion_schedule"]),
        beta_start=float(config["diffusion_beta_start"]),
        beta_end=float(config["diffusion_beta_end"]),
    )


def _get_visual_batch(loader: DataLoader, device: torch.device, min_items: int = 5) -> torch.Tensor:
    fallback: torch.Tensor | None = None
    for raw_batch_data in loader:
        images = raw_batch_data["image"].to(device)
        if fallback is None:
            fallback = images
        if images.shape[0] >= min_items:
            return images

    if fallback is None:
        raise ValueError("No batch could be retrieved from the dataloader.")
    return fallback


def _pick_interpolation_indices(batch_size: int) -> tuple[int, int]:
    if batch_size <= 1:
        raise ValueError("Need at least 2 images in a batch to build interpolation.")
    if batch_size >= 5:
        return 2, 4
    return 0, batch_size - 1


def _interpolate_images(
    model: AutoencoderKL,
    latent_1: torch.Tensor,
    latent_2: torch.Tensor,
    device: torch.device,
    steps: int,
) -> list[np.ndarray]:
    latent_1 = latent_1.to(device)
    latent_2 = latent_2.to(device)

    t_values = torch.linspace(0, 1, steps, device=device)
    latent_interp = torch.stack([torch.lerp(latent_1, latent_2, t).squeeze(0) for t in t_values], dim=0)
    decoded = model.decode(latent_interp)

    return [image.squeeze().detach().cpu().numpy() for image in decoded]


def _save_gan_training_plots(run_info: RunInfo, plots_dir: Path) -> None:
    metrics = run_info.metrics
    config = run_info.hyperparameters["gan_specific"]

    panel_titles = (
        "Training loss curve",
        "Reconstruction metric",
        "KL divergence metric",
        "Perceptual metric",
        "Adversarial metric",
    )
    panel_values = (
        metrics.get("train_generator_loss", []),
        metrics.get("reconstruction_metric", []),
        metrics.get("kld_metric", []),
        metrics.get("perceptual_metric", []),
        metrics.get("adversarial_metric", []),
    )
    save_metric_panels(
        output_path=plots_dir / "GAN - Training Metrics.png",
        panel_titles=panel_titles,
        panel_values=panel_values,
        y_label="Loss",
        figsize=(16, 6),
    )

    discriminator_values = metrics.get("train_discriminator_loss", [])
    adversarial_values = metrics.get("adversarial_metric", [])
    adversarial_weight = float(config.get("adversarial_weight", 1.0))
    if discriminator_values and adversarial_values and adversarial_weight != 0.0:
        count = min(len(discriminator_values), len(adversarial_values))
        x_values = np.arange(1, count + 1)
        generator_curve = [value / adversarial_weight for value in adversarial_values[:count]]
        discriminator_curve = [value / adversarial_weight for value in discriminator_values[:count]]

        save_two_curve_plot(
            output_path=plots_dir / "GAN - Adversarial Training Curves.png",
            x_values=x_values,
            y_values_1=generator_curve,
            y_values_2=discriminator_curve,
            label_1="Generator",
            label_2="Discriminator",
            title="Adversarial Training Curves",
            y_label="Loss",
        )


def _save_ldm_training_plots(run_info: RunInfo, plots_dir: Path) -> None:
    metrics = run_info.metrics
    config = run_info.hyperparameters["ldm_specific"]

    auto_metrics = metrics.get("autoencoder", {})
    diff_metrics = metrics.get("diffusion", {})

    save_metric_panels(
        output_path=plots_dir / "Diffusion Model - Autoencoder Training Metrics.png",
        panel_titles=("Reconstruction", "Generator", "Discriminator"),
        panel_values=(
            auto_metrics.get("epoch_recon_losses", []),
            auto_metrics.get("epoch_gen_losses", []),
            auto_metrics.get("epoch_disc_losses", []),
        ),
        y_label="Loss",
        figsize=(14, 5),
    )

    auto_val_losses = auto_metrics.get("val_recon_losses", [])
    auto_recon_losses = auto_metrics.get("epoch_recon_losses", [])
    auto_val_interval = int(config.get("autoencoder_val_interval", 10))
    if auto_val_losses and auto_recon_losses:
        x_values = np.arange(
            auto_val_interval,
            auto_val_interval * len(auto_val_losses) + 1,
            auto_val_interval,
        )
        train_values = [auto_recon_losses[min(epoch - 1, len(auto_recon_losses) - 1)] for epoch in x_values]
        save_two_curve_plot(
            output_path=plots_dir / "Diffusion Model - Autoencoder Train vs Validation Reconstruction.png",
            x_values=x_values,
            y_values_1=train_values,
            y_values_2=auto_val_losses,
            label_1="Train",
            label_2="Validation",
            title="Autoencoder Reconstruction",
            y_label="Loss",
        )

    gen_losses = auto_metrics.get("epoch_gen_losses", [])
    disc_losses = auto_metrics.get("epoch_disc_losses", [])
    if gen_losses and disc_losses:
        count = min(len(gen_losses), len(disc_losses))
        x_values = np.arange(1, count + 1)
        save_two_curve_plot(
            output_path=plots_dir / "Diffusion Model - Adversarial Training Curves.png",
            x_values=x_values,
            y_values_1=gen_losses[:count],
            y_values_2=disc_losses[:count],
            label_1="Generator",
            label_2="Discriminator",
            title="Adversarial Training Curves",
            y_label="Loss",
        )

    save_metric_panels(
        output_path=plots_dir / "Diffusion Model - Diffusion Training Metrics.png",
        panel_titles=("Diffusion Train Loss",),
        panel_values=(diff_metrics.get("epoch_losses", []),),
        y_label="MSE",
        figsize=(8, 5),
    )

    diffusion_val_losses = diff_metrics.get("val_losses", [])
    diffusion_train_losses = diff_metrics.get("epoch_losses", [])
    diffusion_val_interval = int(config.get("diffusion_val_interval", 40))
    if diffusion_val_losses and diffusion_train_losses:
        x_values = np.arange(
            diffusion_val_interval,
            diffusion_val_interval * len(diffusion_val_losses) + 1,
            diffusion_val_interval,
        )
        train_values = [diffusion_train_losses[min(epoch - 1, len(diffusion_train_losses) - 1)] for epoch in x_values]
        save_two_curve_plot(
            output_path=plots_dir / "Diffusion Model - Diffusion Train vs Validation.png",
            x_values=x_values,
            y_values_1=train_values,
            y_values_2=diffusion_val_losses,
            label_1="Train",
            label_2="Validation",
            title="Diffusion Denoising MSE",
            y_label="MSE",
        )


def _save_latent_space_and_interpolation(
    model_type: str,
    model: AutoencoderKL,
    loader: DataLoader,
    device: torch.device,
    plots_dir: Path,
) -> None:
    title = "GAN Latent Space (t-SNE 2D)" if model_type == "GAN" else "Diffusion Model Latent Space (t-SNE 2D)"
    output_name = (
        "GAN - Latent Space (t-SNE 2D).png" if model_type == "GAN" else "Diffusion Model - Latent Space (t-SNE 2D).png"
    )
    gif_name = "GAN - Latent Interpolation.gif" if model_type == "GAN" else "MedNIST Interpolation.gif"

    latent_vectors = collect_latent_vectors(loader, device, model.encode)
    save_latent_space_plot(
        latent_vectors=latent_vectors,
        output_path=plots_dir / output_name,
        title=title,
    )

    batch_images = _get_visual_batch(loader, device=device, min_items=5)
    index_1, index_2 = _pick_interpolation_indices(batch_images.shape[0])
    with torch.no_grad():
        latent_1, _ = model.encode(batch_images[index_1].unsqueeze(0))
        latent_2, _ = model.encode(batch_images[index_2].unsqueeze(0))
        interpolated_images = _interpolate_images(
            model=model,
            latent_1=latent_1,
            latent_2=latent_2,
            device=device,
            steps=INTERPOLATION_STEPS,
        )

    save_animation_as_gif(
        images=interpolated_images,
        filename=plots_dir / gif_name,
        interval=100,
    )


def _save_ldm_decoded_intermediates(
    run_info: RunInfo,
    autoencoder: AutoencoderKL,
    valid_loader: DataLoader,
    device: torch.device,
    plots_dir: Path,
) -> None:
    if run_info.diffusion_checkpoint_path is None:
        return

    unet = _build_ldm_unet_from_hyperparameters(run_info.hyperparameters, device=device)
    unet_state_dict = torch.load(run_info.diffusion_checkpoint_path, map_location=device)
    unet.load_state_dict(unet_state_dict, strict=True)
    unet.eval()

    scheduler = _build_scheduler_from_hyperparameters(run_info.hyperparameters)
    ldm_config = run_info.hyperparameters["ldm_specific"]
    scheduler.set_timesteps(num_inference_steps=int(ldm_config.get("diffusion_num_inference_steps", 1000)))

    scale_factor = float(run_info.metrics.get("scale_factor", 1.0))
    inferer = LatentDiffusionInferer(scheduler=scheduler, scale_factor=scale_factor)

    batch_images = _get_visual_batch(valid_loader, device=device, min_items=1)
    with torch.no_grad():
        z_mu, z_sigma = autoencoder.encode(batch_images[:1])
        z_sample = autoencoder.sampling(z_mu, z_sigma)

    latent_shape = tuple(z_sample.shape)
    noise = torch.randn(latent_shape, device=device)
    intermediate_steps = max(
        1,
        int(ldm_config.get("diffusion_num_train_timesteps", 1000)) // INTERMEDIATE_DECODE_DIVISOR,
    )

    with torch.no_grad():
        _, intermediates = inferer.sample(
            input_noise=noise,
            diffusion_model=unet,
            scheduler=scheduler,
            save_intermediates=True,
            intermediate_steps=intermediate_steps,
            autoencoder_model=autoencoder,
        )

    if not intermediates:
        raise ValueError("No decoded intermediates were returned by inferer.sample.")

    max_columns = 10
    rows = int(np.ceil(len(intermediates) / max_columns))
    columns = int(np.ceil(len(intermediates) / rows))

    figure, axes = plt.subplots(rows, columns, figsize=(columns * 1.7, rows * 1.7))
    axes_array = np.array(axes, ndmin=1).ravel()

    for index, image in enumerate(intermediates):
        axes_array[index].imshow(image[0, 0].detach().cpu(), vmin=0, vmax=1, cmap="gray")
        axes_array[index].axis("off")

    for index in range(len(intermediates), len(axes_array)):
        axes_array[index].axis("off")

    figure.tight_layout()
    figure.savefig(
        plots_dir / "Diffusion Model - Decoded Intermediates Every 100 Steps.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)


def _save_deterministic_reconstruction_panel(
    model_type: str,
    run_info: RunInfo,
    selected_label: str,
    image_size: int,
    model: AutoencoderKL,
    device: torch.device,
) -> None:
    plots_dir = run_info.run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    reference_images, sample_indices = _get_cached_reference_images(
        selected_label=selected_label,
        image_size=image_size,
    )

    with torch.no_grad():
        latent_mu, _ = model.encode(reference_images.to(device))
        reconstructions = model.decode(latent_mu).detach().cpu().clamp_(0.0, 1.0)

    originals_np = reference_images.squeeze(1).numpy()
    reconstructions_np = reconstructions.squeeze(1).numpy()
    errors_np = np.abs(originals_np - reconstructions_np)
    per_image_mae = errors_np.mean(axis=(1, 2))
    max_error = max(0.1, float(errors_np.max()))

    figure, axes = plt.subplots(3, len(sample_indices), figsize=(1.9 * len(sample_indices), 5.8))
    for column, dataset_index in enumerate(sample_indices):
        axes[0, column].imshow(originals_np[column], cmap="gray", vmin=0.0, vmax=1.0)
        axes[1, column].imshow(reconstructions_np[column], cmap="gray", vmin=0.0, vmax=1.0)
        axes[2, column].imshow(errors_np[column], cmap="magma", vmin=0.0, vmax=max_error)

        axes[0, column].set_title(f"idx {dataset_index}")
        axes[1, column].set_title(f"MAE {per_image_mae[column]:.4f}")

        for row in range(3):
            axes[row, column].axis("off")

    axes[0, 0].set_ylabel("Original (x)", fontsize=11)
    axes[1, 0].set_ylabel("Reconstruction (x_hat)", fontsize=11)
    axes[2, 0].set_ylabel("Abs Error |x - x_hat|", fontsize=11)

    score = float("nan")
    if model_type == "GAN":
        score = float(run_info.metrics.get("best_valid_metric", float("nan")))
    elif model_type == "LDM":
        val_losses = run_info.metrics.get("autoencoder", {}).get("val_recon_losses", [])
        if val_losses:
            score = float(min(val_losses))

    figure.suptitle(
        f"{model_type} deterministic reconstructions\n"
        f"run={run_info.run_dir.name} | score={score:.6f} | label={selected_label} | mean MAE={per_image_mae.mean():.4f}",
        fontsize=13,
    )
    figure.text(
        0.5,
        0.01,
        "Row 3 visualizes per-pixel absolute error |x - x_hat|. Brighter values indicate larger reconstruction error.",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout()

    output_path = plots_dir / f"{model_type} - Deterministic Reference Reconstructions.png"
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)

    summary = {
        "model_type": model_type,
        "run_dir": str(run_info.run_dir),
        "checkpoint_path": str(run_info.auto_checkpoint_path),
        "selection_seed": SEED,
        "selected_label": selected_label,
        "sample_indices": sample_indices,
        "selection_dataset": "MedNIST validation section with deterministic no-cache preprocessing",
        "panel_rows": {
            "row_1": "Original (x)",
            "row_2": "Reconstruction (x_hat)",
            "row_3": "Absolute error map |x - x_hat|",
        },
        "score": score,
        "mean_mae": float(per_image_mae.mean()),
        "per_image_mae": [float(value) for value in per_image_mae],
    }
    summary_path = plots_dir / f"{model_type} - Deterministic Reference Reconstructions.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _regenerate_gan_run(run_info: RunInfo, device: torch.device) -> None:
    selected_label, batch_size, image_size, train_valid_ratio = _resolve_run_common(run_info.hyperparameters)
    _, _, test_loader = _get_cached_dataloaders(
        selected_label=selected_label,
        batch_size=batch_size,
        image_size=image_size,
        train_valid_ratio=train_valid_ratio,
    )

    model = _load_autoencoder_for_run(run_info, device=device)
    plots_dir = run_info.run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    _save_gan_training_plots(run_info, plots_dir)
    _save_latent_space_and_interpolation(
        model_type="GAN",
        model=model,
        loader=test_loader,
        device=device,
        plots_dir=plots_dir,
    )
    _save_deterministic_reconstruction_panel(
        model_type="GAN",
        run_info=run_info,
        selected_label=selected_label,
        image_size=image_size,
        model=model,
        device=device,
    )


def _regenerate_ldm_run(run_info: RunInfo, device: torch.device) -> None:
    selected_label, batch_size, image_size, train_valid_ratio = _resolve_run_common(run_info.hyperparameters)
    _, valid_loader, _ = _get_cached_dataloaders(
        selected_label=selected_label,
        batch_size=batch_size,
        image_size=image_size,
        train_valid_ratio=train_valid_ratio,
    )

    autoencoder = _load_autoencoder_for_run(run_info, device=device)
    plots_dir = run_info.run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    _save_ldm_training_plots(run_info, plots_dir)
    _save_latent_space_and_interpolation(
        model_type="LDM",
        model=autoencoder,
        loader=valid_loader,
        device=device,
        plots_dir=plots_dir,
    )
    _save_ldm_decoded_intermediates(
        run_info=run_info,
        autoencoder=autoencoder,
        valid_loader=valid_loader,
        device=device,
        plots_dir=plots_dir,
    )
    _save_deterministic_reconstruction_panel(
        model_type="LDM",
        run_info=run_info,
        selected_label=selected_label,
        image_size=image_size,
        model=autoencoder,
        device=device,
    )


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_runs = 0
    failed_runs: list[str] = []

    for model_type in ("GAN", "LDM"):
        runs = _collect_runs(model_type)
        if not runs:
            print(f"[skip] {model_type}: no loadable run found.")
            continue

        print(f"[info] {model_type}: regenerating artifacts for {len(runs)} runs.")
        for run_info in runs:
            try:
                if model_type == "GAN":
                    _regenerate_gan_run(run_info, device=device)
                else:
                    _regenerate_ldm_run(run_info, device=device)

                total_runs += 1
                print(f"[saved] {model_type} {run_info.run_dir.name}")
            except Exception as exc:  # noqa: BLE001
                failed_runs.append(f"{model_type} {run_info.run_dir.name}: {exc}")
                print(f"[skip] {model_type} {run_info.run_dir.name}: {exc}")

    print(f"[done] regenerated artifacts for {total_runs} runs.")
    if failed_runs:
        print("[warn] failed runs:")
        for failed in failed_runs:
            print(f"  - {failed}")


if __name__ == "__main__":
    main()
