import contextlib
import time
from typing import Any

import torch
import torch.nn as nn
from monai.data import DataLoader
from monai.losses import PatchAdversarialLoss, PerceptualLoss
from monai.networks.layers import Act
from monai.networks.nets import AutoencoderKL, PatchDiscriminator
from tqdm.auto import tqdm
from torchinfo import summary

from common import (
    BatchData,
    build_alex_perceptual_loss,
    get_mednist_dataloaders,
    print_library_versions,
)
from settings import (
    ADVERSARIAL_WEIGHT,
    ATTENTION_LEVELS,
    AUTOENCODER_CHANNELS,
    AUTOENCODER_LEARNING_RATE,
    BATCH_SIZE,
    DISCRIMINATOR_CHANNELS,
    DISCRIMINATOR_LEARNING_RATE,
    DISCRIMINATOR_NUM_LAYERS_D,
    IMAGE_SIZE,
    IN_CHANNELS,
    KL_WEIGHT,
    LATENT_CHANNELS,
    NUM_EPOCHS,
    NUM_RES_BLOCKS,
    OUT_CHANNELS,
    PERCEPTUAL_WEIGHT,
    SELECTED_LABEL,
    SPATIAL_DIMS,
    TRAIN_VALID_RATIO,
    WITH_DECODER_NONLOCAL_ATTN,
    WITH_ENCODER_NONLOCAL_ATTN,
)
from utils import create_run_directory, ensure_model_type_directories, save_json, pgcd

# GroupNorm groups used inside the model.
# Using the first channel count is MONAI's common stable default for this setup.
NORM_NUM_GROUPS = pgcd(*AUTOENCODER_CHANNELS)

# "Hidden layers" of the discriminator
# Doubles for each block until the last, where it is squashed back to 1 (1x6x6 for 36 patches)
SAVE_BEST_MODEL_FROM_METRIC = True


class GANComponents:
    @staticmethod
    def build_generator(device: torch.device) -> AutoencoderKL:
        with contextlib.redirect_stdout(None):
            model = AutoencoderKL(
                spatial_dims=SPATIAL_DIMS,
                in_channels=IN_CHANNELS,
                out_channels=OUT_CHANNELS,
                channels=AUTOENCODER_CHANNELS,
                latent_channels=LATENT_CHANNELS,
                num_res_blocks=NUM_RES_BLOCKS,
                norm_num_groups=NORM_NUM_GROUPS,
                attention_levels=ATTENTION_LEVELS,
                with_encoder_nonlocal_attn=WITH_ENCODER_NONLOCAL_ATTN,
                with_decoder_nonlocal_attn=WITH_DECODER_NONLOCAL_ATTN,
            )

        model.to(device)
        return model

    @staticmethod
    def build_discriminator(device: torch.device) -> PatchDiscriminator:
        with contextlib.redirect_stdout(None):
            discriminator = PatchDiscriminator(
                spatial_dims=SPATIAL_DIMS,
                num_layers_d=DISCRIMINATOR_NUM_LAYERS_D,
                channels=DISCRIMINATOR_CHANNELS,
                in_channels=IN_CHANNELS,
                out_channels=OUT_CHANNELS,
                activation=(Act.LEAKYRELU, {"negative_slope": 0.2}),
                norm="BATCH",
                bias=False,
                padding=1,
            )

        discriminator.to(device)
        return discriminator

    @staticmethod
    def build_losses(device: torch.device) -> tuple[PerceptualLoss, PatchAdversarialLoss, nn.L1Loss]:
        perceptual_loss_fn = build_alex_perceptual_loss(device)
        adversarial_loss_fn = PatchAdversarialLoss(criterion="least_squares")
        reconstruction_loss_fn = nn.L1Loss()
        reconstruction_loss_fn.to(device)

        return perceptual_loss_fn, adversarial_loss_fn, reconstruction_loss_fn


class GANMath:
    @staticmethod
    def vae_gaussian_kl_loss(mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        # L_KL = 0.5 * sum(mu^2 + sigma^2 - log(sigma^2) - 1)
        kl_loss = 0.5 * torch.sum(mu.pow(2) + sigma.pow(2) - torch.log(sigma.pow(2)) - 1, dim=[1, 2, 3])
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
        return kl_loss

    @staticmethod
    def generator_loss(
        recon_x: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        logits: torch.Tensor,
        perceptual_loss_fn: PerceptualLoss,
        adversarial_loss_fn: PatchAdversarialLoss,
        l1_loss_fn: nn.L1Loss,
    ) -> tuple[torch.Tensor, float, float, float, float]:
        recon_loss = l1_loss_fn(recon_x.float(), x.float())
        kl_loss = GANMath.vae_gaussian_kl_loss(mu, sigma)
        perceptual_loss_value = perceptual_loss_fn(recon_x.float(), x.float())
        adversarial_loss_value = adversarial_loss_fn(logits, target_is_real=True, for_discriminator=False)

        # L_G = L_recon + λ_kl * L_kl + λ_p * L_perceptual + λ_adv * L_adv
        total_loss = (
            recon_loss
            + KL_WEIGHT * kl_loss
            + PERCEPTUAL_WEIGHT * perceptual_loss_value
            + ADVERSARIAL_WEIGHT * adversarial_loss_value
        )

        return (
            total_loss,
            recon_loss.item(),
            (KL_WEIGHT * kl_loss).item(),
            (PERCEPTUAL_WEIGHT * perceptual_loss_value).item(),
            (ADVERSARIAL_WEIGHT * adversarial_loss_value).item(),
        )


class GANTraining:
    @staticmethod
    def train(
        model: AutoencoderKL,
        discriminator: PatchDiscriminator,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        optimizer_generator: torch.optim.Optimizer,
        optimizer_discriminator: torch.optim.Optimizer,
        perceptual_loss_fn: PerceptualLoss,
        adversarial_loss_fn: PatchAdversarialLoss,
        l1_loss_fn: nn.L1Loss,
        device: torch.device,
    ) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
        train_generator_loss_list: list[float] = []
        train_discriminator_loss_list: list[float] = []
        valid_metric_list: list[float] = []
        reconstruction_metric_list: list[float] = []
        kld_metric_list: list[float] = []
        perceptual_metric_list: list[float] = []
        adversarial_metric_list: list[float] = []

        best_valid_metric = float("inf")
        best_model: dict[str, torch.Tensor] | None = None
        best_epoch = 0

        for epoch in range(NUM_EPOCHS):
            model.train()
            discriminator.train()

            train_generator_loss = 0.0
            train_discriminator_loss = 0.0
            reconstruction_metric = 0.0
            kld_metric = 0.0
            perceptual_metric = 0.0
            adversarial_metric = 0.0

            seen_samples = 0
            progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=110)
            progress_bar.set_description(f"Epoch {epoch}")

            for step, raw_batch_data in progress_bar:
                batch_data: BatchData = raw_batch_data
                inputs = batch_data["image"].to(device)
                seen_samples += inputs.size(0)

                optimizer_generator.zero_grad()

                reconstruction, z_mu, z_sigma = model(inputs)
                logits_fake = discriminator(reconstruction.contiguous().float())[-1]

                (
                    loss_generator,
                    reconstruction_val,
                    kld_val,
                    perceptual_val,
                    adversarial_val,
                ) = GANMath.generator_loss(
                    reconstruction,
                    inputs,
                    z_mu,
                    z_sigma,
                    logits_fake,
                    perceptual_loss_fn,
                    adversarial_loss_fn,
                    l1_loss_fn,
                )

                loss_generator.backward()
                optimizer_generator.step()

                train_generator_loss += loss_generator.item() * inputs.size(0)
                reconstruction_metric += reconstruction_val * inputs.size(0)
                kld_metric += kld_val * inputs.size(0)
                perceptual_metric += perceptual_val * inputs.size(0)
                adversarial_metric += adversarial_val * inputs.size(0)

                if ADVERSARIAL_WEIGHT > 0:
                    optimizer_discriminator.zero_grad(set_to_none=True)

                    logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
                    logits_real = discriminator(inputs.contiguous().detach())[-1]

                    loss_d_fake = adversarial_loss_fn(logits_fake, target_is_real=False, for_discriminator=True)
                    loss_d_real = adversarial_loss_fn(logits_real, target_is_real=True, for_discriminator=True)

                    # L_D = λ_adv * 0.5 * (L_fake + L_real)
                    loss_discriminator = ADVERSARIAL_WEIGHT * 0.5 * (loss_d_fake + loss_d_real)

                    loss_discriminator.backward()
                    optimizer_discriminator.step()

                    train_discriminator_loss += loss_discriminator.item() * inputs.size(0)

                progress_bar.set_postfix(
                    {
                        "recons_loss": reconstruction_metric / seen_samples,
                        "gen_loss": train_generator_loss / seen_samples,
                        "disc_loss": (train_discriminator_loss / seen_samples) if ADVERSARIAL_WEIGHT > 0 else 0.0,
                    }
                )

            progress_bar.close()

            train_generator_loss_list.append(train_generator_loss / len(train_loader.dataset))
            reconstruction_metric_list.append(reconstruction_metric / len(train_loader.dataset))
            kld_metric_list.append(kld_metric / len(train_loader.dataset))
            perceptual_metric_list.append(perceptual_metric / len(train_loader.dataset))
            adversarial_metric_list.append(adversarial_metric / len(train_loader.dataset))

            if ADVERSARIAL_WEIGHT > 0:
                train_discriminator_loss_list.append(train_discriminator_loss / len(train_loader.dataset))

            model.eval()
            valid_metric = 0.0

            with torch.no_grad():
                for raw_batch_data in valid_loader:
                    batch_data: BatchData = raw_batch_data
                    inputs = batch_data["image"].to(device)

                    reconstruction, _, _ = model(inputs)
                    recon_val = l1_loss_fn(reconstruction.float(), inputs.float())
                    valid_metric += recon_val.item() * inputs.size(0)

            valid_metric_list.append(valid_metric / len(valid_loader.dataset))

            print(
                f"Epoch: {epoch+1} \tTraining Loss: {train_generator_loss_list[-1]:.6f} \tValidation metric: {valid_metric_list[-1]:.6f}"
            )

            if SAVE_BEST_MODEL_FROM_METRIC:
                if valid_metric_list[-1] < best_valid_metric:
                    best_valid_metric = valid_metric_list[-1]
                    best_model = model.state_dict()
                    best_epoch = epoch + 1
            else:
                best_valid_metric = valid_metric_list[-1]
                best_model = model.state_dict()
                best_epoch = epoch + 1

        if best_model is None:
            raise RuntimeError("Best model was not computed.")

        metrics = {
            "best_epoch": best_epoch,
            "best_valid_metric": best_valid_metric,
            "train_generator_loss": train_generator_loss_list,
            "train_discriminator_loss": train_discriminator_loss_list,
            "valid_metric": valid_metric_list,
            "reconstruction_metric": reconstruction_metric_list,
            "kld_metric": kld_metric_list,
            "perceptual_metric": perceptual_metric_list,
            "adversarial_metric": adversarial_metric_list,
        }

        return metrics, best_model

    @staticmethod
    def evaluate_test_reconstruction(
        model: AutoencoderKL,
        test_loader: DataLoader,
        l1_loss_fn: nn.L1Loss,
        device: torch.device,
    ) -> float:
        test_metric = 0.0
        model.eval()

        with torch.no_grad():
            for raw_batch_data in test_loader:
                batch_data: BatchData = raw_batch_data
                inputs = batch_data["image"].to(device)

                reconstruction, _, _ = model(inputs)
                recon_val = l1_loss_fn(reconstruction.float(), inputs.float())
                test_metric += recon_val.item() * inputs.size(0)

        return test_metric / len(test_loader.dataset)


if __name__ == "__main__":
    print_library_versions()

    ensure_model_type_directories()
    run_dir = create_run_directory("GAN")
    plots_dir = run_dir / "plots"
    models_dir = run_dir / "models"
    plots_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    run_start_time = time.perf_counter()

    train_loader, valid_loader, test_loader = get_mednist_dataloaders()

    print(f"Training dataset size: {len(train_loader.dataset)}")
    print(f"Validation dataset size: {len(valid_loader.dataset)}")
    print(f"Test dataset size: {len(test_loader.dataset)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GANComponents.build_generator(device)
    discriminator = GANComponents.build_discriminator(device)

    summary_kwargs = dict(col_names=["input_size", "output_size", "num_params"], depth=3, verbose=0)
    summary(discriminator, (1, 1, IMAGE_SIZE, IMAGE_SIZE), device="cpu", **summary_kwargs)
    discriminator.to(device)

    perceptual_loss_fn, adversarial_loss_fn, l1_loss_fn = GANComponents.build_losses(device)

    optimizer_generator = torch.optim.Adam(model.parameters(), lr=AUTOENCODER_LEARNING_RATE)
    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=DISCRIMINATOR_LEARNING_RATE)

    hyperparameters = {
        "model_type": "GAN",
        "common": {
            "epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "image_size": IMAGE_SIZE,
            "train_valid_ratio": TRAIN_VALID_RATIO,
            "selected_label": SELECTED_LABEL,
        },
        "gan_specific": {
            "spatial_dims": SPATIAL_DIMS,
            "in_channels": IN_CHANNELS,
            "out_channels": OUT_CHANNELS,
            "channels": AUTOENCODER_CHANNELS,
            "latent_channels": LATENT_CHANNELS,
            "num_res_blocks": NUM_RES_BLOCKS,
            "norm_num_groups": NORM_NUM_GROUPS,
            "attention_levels": ATTENTION_LEVELS,
            "num_layers_d": DISCRIMINATOR_NUM_LAYERS_D,
            "discriminator_channels": DISCRIMINATOR_CHANNELS,
            "learning_rate_g": AUTOENCODER_LEARNING_RATE,
            "learning_rate_d": DISCRIMINATOR_LEARNING_RATE,
            "kl_weight": KL_WEIGHT,
            "perceptual_weight": PERCEPTUAL_WEIGHT,
            "adversarial_weight": ADVERSARIAL_WEIGHT,
        },
    }

    save_json(hyperparameters, run_dir / "hyperparameters.json")

    training_metrics, best_model = GANTraining.train(
        model=model,
        discriminator=discriminator,
        train_loader=train_loader,
        valid_loader=valid_loader,
        optimizer_generator=optimizer_generator,
        optimizer_discriminator=optimizer_discriminator,
        perceptual_loss_fn=perceptual_loss_fn,
        adversarial_loss_fn=adversarial_loss_fn,
        l1_loss_fn=l1_loss_fn,
        device=device,
    )

    model.load_state_dict(best_model)
    torch.save(best_model, models_dir / "best_test_lossmodel.pth")

    print(
        f"Best model selected at epoch {training_metrics['best_epoch']} with validation loss: {training_metrics['best_valid_metric']:.6f}"
    )

    test_metric = GANTraining.evaluate_test_reconstruction(
        model=model,
        test_loader=test_loader,
        l1_loss_fn=l1_loss_fn,
        device=device,
    )

    print("Test reconstruction metric: {:.6f}\n".format(test_metric))

    metrics = dict(training_metrics)
    metrics["test_reconstruction_metric"] = test_metric
    metrics["run_duration_seconds"] = time.perf_counter() - run_start_time
    save_json(metrics, run_dir / "metrics.json")
