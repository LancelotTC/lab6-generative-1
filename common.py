import os
import warnings
from pathlib import Path
from typing import Callable, Sequence

import matplotlib.pyplot as plt
import monai
import numpy as np
import torch
import torchvision
from monai.apps import MedNISTDataset
from monai.data import DataLoader, Dataset
from monai.losses import PerceptualLoss
from monai.transforms import Compose, EnsureChannelFirstd, LoadImaged, RandAffined, Resized, ScaleIntensityRanged
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import random_split

from settings import (
    BATCH_SIZE,
    COMMON_TSNE_RANDOM_STATE,
    DATA_ROOT_DIR,
    IMAGE_SIZE,
    MAX_INTENSITY_VALUE,
    MEDNIST_DATA_DIR,
    MEDNIST_TRAIN_SECTION,
    MEDNIST_VALID_SECTION,
    MIN_INTENSITY_VALUE,
    NUM_WORKERS,
    SEED,
    SELECTED_LABEL,
    TRAIN_VALID_RATIO,
)

BatchData = dict[str, torch.Tensor | list[str]]


def print_library_versions() -> None:
    print(f"PyTorch version: {torch.__version__}")
    print(f"Torchvision version: {torchvision.__version__}")
    print(f"Numpy version: {np.__version__}")
    print(f"Monai version: {monai.__version__}")


def get_mednist_dataloaders(
    batch_size: int = BATCH_SIZE,
    image_size: int = IMAGE_SIZE,
    train_valid_ratio: float = TRAIN_VALID_RATIO,
    selected_label: str = SELECTED_LABEL,
    num_workers: int = NUM_WORKERS,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    download = not os.path.exists(MEDNIST_DATA_DIR)
    train_set = MedNISTDataset(root_dir=DATA_ROOT_DIR, section=MEDNIST_TRAIN_SECTION, download=download, seed=SEED)
    test_set = MedNISTDataset(root_dir=DATA_ROOT_DIR, section=MEDNIST_VALID_SECTION, download=download, seed=SEED)

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
            ),
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

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )
    return train_loader, valid_loader, test_loader


def save_metric_panels(
    output_path: Path,
    panel_titles: Sequence[str],
    panel_values: Sequence[Sequence[float]],
    y_label: str = "Loss",
    figsize: tuple[int, int] = (16, 6),
) -> None:
    panel_count = len(panel_titles)

    plt.figure(figsize=figsize)

    for panel_index, (title, values) in enumerate(zip(panel_titles, panel_values), start=1):
        plt.subplot(1, panel_count, panel_index)
        plt.plot(values, color="C0", linewidth=2.0, label=title)
        plt.title(title)
        plt.xlabel("Epochs")
        plt.ylabel(y_label)
        plt.legend()

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_two_curve_plot(
    output_path: Path,
    x_values: Sequence[float] | np.ndarray,
    y_values_1: Sequence[float] | np.ndarray,
    y_values_2: Sequence[float] | np.ndarray,
    label_1: str,
    label_2: str,
    title: str,
    y_label: str,
) -> None:
    plt.figure(figsize=(8, 6))
    plt.title(title, fontsize=20)

    plt.plot(x_values, y_values_1, color="C0", linewidth=2.0, label=label_1)
    plt.plot(x_values, y_values_2, color="C1", linewidth=2.0, label=label_2)

    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    plt.xlabel("Epochs", fontsize=16)
    plt.ylabel(y_label, fontsize=16)
    plt.legend(prop={"size": 14})

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def build_alex_perceptual_loss(device: torch.device) -> PerceptualLoss:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Arguments other than a weight enum or `None` for 'weights' are deprecated.*",
            category=UserWarning,
            module="torchvision.models._utils",
        )
        perceptual_loss_fn = PerceptualLoss(spatial_dims=2, network_type="alex")

    perceptual_loss_fn.to(device)
    return perceptual_loss_fn


def collect_latent_vectors(
    data_loader: DataLoader,
    device: torch.device,
    encode_fn: Callable[[torch.Tensor], torch.Tensor | tuple[torch.Tensor, ...]],
) -> np.ndarray:
    latent_vectors: list[np.ndarray] = []

    with torch.no_grad():
        for raw_batch_data in data_loader:
            batch_data: BatchData = raw_batch_data
            inputs = batch_data["image"].to(device)

            encoded = encode_fn(inputs)
            latent_tensor = encoded[0] if isinstance(encoded, tuple) else encoded
            latent_vectors.append(latent_tensor.detach().cpu().reshape(latent_tensor.shape[0], -1).numpy())

    return np.concatenate(latent_vectors, axis=0)


def save_latent_space_plot(
    latent_vectors: np.ndarray,
    output_path: Path,
    title: str,
    random_state: int = COMMON_TSNE_RANDOM_STATE,
) -> np.ndarray:
    if latent_vectors.ndim != 2:
        raise ValueError("latent_vectors must be a 2D array of shape [n_samples, n_features].")

    if latent_vectors.shape[0] < 2:
        raise ValueError("At least 2 samples are required to build a 2D latent-space embedding.")

    vectors = latent_vectors.astype(np.float32, copy=False)
    vectors = (vectors - vectors.mean(axis=0, keepdims=True)) / (vectors.std(axis=0, keepdims=True) + 1e-8)

    if vectors.shape[1] > 50:
        pca = PCA(n_components=min(50, vectors.shape[0], vectors.shape[1]), random_state=random_state)
        vectors = pca.fit_transform(vectors)

    perplexity = min(30, max(5, (vectors.shape[0] - 1) // 3))
    perplexity = min(perplexity, vectors.shape[0] - 1)

    tsne = TSNE(
        n_components=2,
        random_state=random_state,
        init="pca",
        learning_rate="auto",
        perplexity=perplexity,
    )
    embedding_2d = tsne.fit_transform(vectors)

    plt.figure(figsize=(8, 6))
    plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], s=10, alpha=0.75, c="C0", edgecolors="none")
    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")

    x_coords = embedding_2d[:, 0]
    y_coords = embedding_2d[:, 1]
    min_axis = min(float(x_coords.min()), float(y_coords.min()))
    max_axis = max(float(x_coords.max()), float(y_coords.max()))
    axis_span = max_axis - min_axis
    axis_padding = 0.05 * axis_span if axis_span > 0 else 1.0
    axis_min = min_axis - axis_padding
    axis_max = max_axis + axis_padding

    ax = plt.gca()
    ax.set_xlim(axis_min, axis_max)
    ax.set_ylim(axis_min, axis_max)
    ax.set_aspect("equal", adjustable="box")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    return embedding_2d
