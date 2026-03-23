import contextlib
import time

import torch
import torch.nn.functional as F
from monai.data import DataLoader
from monai.inferers import LatentDiffusionInferer
from monai.losses import PatchAdversarialLoss, PerceptualLoss
from monai.networks.nets import AutoencoderKL, DiffusionModelUNet, PatchDiscriminator
from monai.networks.schedulers import DDPMScheduler as LDMScheduler
from monai.utils import first
from torch.amp import GradScaler, autocast
from torchinfo import summary
from tqdm.auto import tqdm

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

NORM_NUM_GROUPS = pgcd(*AUTOENCODER_CHANNELS)
DIFFUSION_LEARNING_RATE = 1e-4

AUTOENCODER_MAX_EPOCHS = 60
AUTOENCODER_VAL_INTERVAL = 10
AUTOENCODER_WARM_UP_N_EPOCHS = 10

DIFFUSION_MAX_EPOCHS = 80
DIFFUSION_VAL_INTERVAL = 40

DIFFUSION_CHANNELS = (128, 256, 512)
DIFFUSION_ATTENTION_LEVELS = (False, True, True)
DIFFUSION_NUM_HEAD_CHANNELS = (0, 256, 512)
DIFFUSION_NORM_NUM_GROUPS = pgcd(*DIFFUSION_CHANNELS)
DIFFUSION_NUM_TRAIN_TIMESTEPS = 1000
DIFFUSION_BETA_START = 0.0015
DIFFUSION_BETA_END = 0.0195
DIFFUSION_SCHEDULE = "linear_beta"
DIFFUSION_NUM_INFERENCE_STEPS = 1000


class LDMComponents:
    @staticmethod
    def build_autoencoder(device: torch.device) -> AutoencoderKL:
        with contextlib.redirect_stdout(None):
            autoencoder = AutoencoderKL(
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

        autoencoder.to(device)
        return autoencoder

    @staticmethod
    def build_discriminator(device: torch.device) -> PatchDiscriminator:
        discriminator = PatchDiscriminator(
            spatial_dims=SPATIAL_DIMS,
            num_layers_d=DISCRIMINATOR_NUM_LAYERS_D,
            channels=DISCRIMINATOR_CHANNELS,
            in_channels=IN_CHANNELS,
            out_channels=OUT_CHANNELS,
        )
        discriminator.to(device)
        return discriminator

    @staticmethod
    def build_diffusion_unet(device: torch.device) -> DiffusionModelUNet:
        diffusion_unet = DiffusionModelUNet(
            spatial_dims=SPATIAL_DIMS,
            in_channels=LATENT_CHANNELS,
            out_channels=LATENT_CHANNELS,
            num_res_blocks=NUM_RES_BLOCKS,
            channels=DIFFUSION_CHANNELS,
            attention_levels=DIFFUSION_ATTENTION_LEVELS,
            num_head_channels=DIFFUSION_NUM_HEAD_CHANNELS,
            norm_num_groups=DIFFUSION_NORM_NUM_GROUPS,
        )
        diffusion_unet.to(device)
        return diffusion_unet

    @staticmethod
    def build_scheduler() -> LDMScheduler:
        return LDMScheduler(
            num_train_timesteps=DIFFUSION_NUM_TRAIN_TIMESTEPS,
            schedule=DIFFUSION_SCHEDULE,
            beta_start=DIFFUSION_BETA_START,
            beta_end=DIFFUSION_BETA_END,
        )

    @staticmethod
    def summarize_unet(unet: DiffusionModelUNet) -> None:
        class WrappedUNet(torch.nn.Module):
            def __init__(self, model: DiffusionModelUNet, timestep: int = 10) -> None:
                super().__init__()
                self.model = model
                self.timestep = timestep

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                timesteps = torch.full((x.size(0),), self.timestep, device=x.device, dtype=torch.long)
                return self.model(x, timesteps)

        summary_kwargs = dict(col_names=["input_size", "output_size", "num_params"], depth=3, verbose=0)
        wrapped = WrappedUNet(unet)
        summary(wrapped, (1, LATENT_CHANNELS, 16, 16), device="cpu", **summary_kwargs)


class LDMMath:
    @staticmethod
    def vae_gaussian_kl_loss(mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        # L_KL = 0.5 * sum(mu^2 + sigma^2 - log(sigma^2) - 1)
        kl_loss = 0.5 * torch.sum(mu.pow(2) + sigma.pow(2) - torch.log(sigma.pow(2)) - 1, dim=[1, 2, 3])
        return torch.sum(kl_loss) / kl_loss.shape[0]


class LDMTraining:
    @staticmethod
    def train_autoencoder(
        autoencoder: AutoencoderKL,
        discriminator: PatchDiscriminator,
        perceptual_loss_fn: PerceptualLoss,
        adversarial_loss_fn: PatchAdversarialLoss,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        optimizer_g: torch.optim.Optimizer,
        optimizer_d: torch.optim.Optimizer,
        scaler_g: GradScaler,
        scaler_d: GradScaler,
        device: torch.device,
    ) -> dict[str, list[float]]:
        amp_enabled = device.type == "cuda"

        epoch_recon_losses: list[float] = []
        epoch_gen_losses: list[float] = []
        epoch_disc_losses: list[float] = []
        val_recon_losses: list[float] = []

        for epoch in range(AUTOENCODER_MAX_EPOCHS):
            autoencoder.train()
            discriminator.train()

            epoch_loss = 0.0
            gen_epoch_loss = 0.0
            disc_epoch_loss = 0.0

            progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=110)
            progress_bar.set_description(f"Autoencoder Epoch {epoch}")

            for step, raw_batch_data in progress_bar:
                batch_data: BatchData = raw_batch_data
                images = batch_data["image"].to(device)

                optimizer_g.zero_grad(set_to_none=True)

                with autocast(device.type, enabled=amp_enabled):
                    reconstruction, z_mu, z_sigma = autoencoder(images)

                    recons_loss = F.l1_loss(reconstruction.float(), images.float())
                    p_loss = perceptual_loss_fn(reconstruction.float(), images.float())
                    kl_loss = LDMMath.vae_gaussian_kl_loss(z_mu, z_sigma)

                    # L_AE = L_recon + lambda_kl * L_kl + lambda_perc * L_perceptual
                    loss_g = recons_loss + (KL_WEIGHT * kl_loss) + (PERCEPTUAL_WEIGHT * p_loss)

                    generator_loss = torch.tensor(0.0, device=device)
                    if epoch > AUTOENCODER_WARM_UP_N_EPOCHS:
                        logits_fake = discriminator(reconstruction.contiguous().float())[-1]
                        generator_loss = adversarial_loss_fn(
                            logits_fake,
                            target_is_real=True,
                            for_discriminator=False,
                        )
                        loss_g = loss_g + (ADVERSARIAL_WEIGHT * generator_loss)

                scaler_g.scale(loss_g).backward()
                scaler_g.step(optimizer_g)
                scaler_g.update()

                discriminator_loss = torch.tensor(0.0, device=device)
                if epoch > AUTOENCODER_WARM_UP_N_EPOCHS:
                    optimizer_d.zero_grad(set_to_none=True)

                    with autocast(device.type, enabled=amp_enabled):
                        logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
                        loss_d_fake = adversarial_loss_fn(logits_fake, target_is_real=False, for_discriminator=True)

                        logits_real = discriminator(images.contiguous().detach())[-1]
                        loss_d_real = adversarial_loss_fn(logits_real, target_is_real=True, for_discriminator=True)

                        # L_D = lambda_adv * 0.5 * (L_fake + L_real)
                        discriminator_loss = 0.5 * (loss_d_fake + loss_d_real)
                        loss_d = ADVERSARIAL_WEIGHT * discriminator_loss

                    scaler_d.scale(loss_d).backward()
                    scaler_d.step(optimizer_d)
                    scaler_d.update()

                epoch_loss += recons_loss.item()
                if epoch > AUTOENCODER_WARM_UP_N_EPOCHS:
                    gen_epoch_loss += generator_loss.item()
                    disc_epoch_loss += discriminator_loss.item()

                progress_bar.set_postfix(
                    {
                        "recons_loss": epoch_loss / (step + 1),
                        "gen_loss": gen_epoch_loss / (step + 1),
                        "disc_loss": disc_epoch_loss / (step + 1),
                    }
                )

            epoch_recon_losses.append(epoch_loss / (step + 1))
            epoch_gen_losses.append(gen_epoch_loss / (step + 1))
            epoch_disc_losses.append(disc_epoch_loss / (step + 1))

            if (epoch + 1) % AUTOENCODER_VAL_INTERVAL == 0:
                autoencoder.eval()
                val_loss = 0.0

                with torch.no_grad():
                    for val_step, raw_batch_data in enumerate(valid_loader, start=1):
                        batch_data: BatchData = raw_batch_data
                        images = batch_data["image"].to(device)

                        with autocast(device.type, enabled=amp_enabled):
                            reconstruction, _, _ = autoencoder(images)
                            recons_loss = F.l1_loss(images.float(), reconstruction.float())

                        val_loss += recons_loss.item()

                val_loss /= val_step
                val_recon_losses.append(val_loss)
                print(f"Autoencoder epoch {epoch + 1} val loss: {val_loss:.4f}")

        return {
            "epoch_recon_losses": epoch_recon_losses,
            "epoch_gen_losses": epoch_gen_losses,
            "epoch_disc_losses": epoch_disc_losses,
            "val_recon_losses": val_recon_losses,
        }

    @staticmethod
    def compute_scale_factor(
        autoencoder: AutoencoderKL,
        train_loader: DataLoader,
        device: torch.device,
    ) -> float:
        amp_enabled = device.type == "cuda"
        check_data: BatchData = first(train_loader)

        with torch.no_grad():
            with autocast(device.type, enabled=amp_enabled):
                z = autoencoder.encode_stage_2_inputs(check_data["image"].to(device))

        scale_factor = float((1 / torch.std(z)).item())
        print(f"Scaling factor set to {scale_factor}")
        return scale_factor

    @staticmethod
    def train_diffusion(
        autoencoder: AutoencoderKL,
        unet: DiffusionModelUNet,
        inferer: LatentDiffusionInferer,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scaler: GradScaler,
        device: torch.device,
    ) -> dict[str, list[float]]:
        amp_enabled = device.type == "cuda"

        epoch_losses: list[float] = []
        val_losses: list[float] = []

        for epoch in range(DIFFUSION_MAX_EPOCHS):
            unet.train()
            autoencoder.eval()

            epoch_loss = 0.0
            progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=90)
            progress_bar.set_description(f"Diffusion Epoch {epoch}")

            for step, raw_batch_data in progress_bar:
                batch_data: BatchData = raw_batch_data
                images = batch_data["image"].to(device)

                optimizer.zero_grad(set_to_none=True)

                with autocast(device.type, enabled=amp_enabled):
                    z_mu, z_sigma = autoencoder.encode(images)
                    z = autoencoder.sampling(z_mu, z_sigma)

                    noise = torch.randn_like(z)
                    timesteps = torch.randint(
                        0,
                        inferer.scheduler.num_train_timesteps,
                        (z.shape[0],),
                        device=z.device,
                    ).long()

                    noise_pred = inferer(
                        inputs=images,
                        diffusion_model=unet,
                        noise=noise,
                        timesteps=timesteps,
                        autoencoder_model=autoencoder,
                    )

                    # L_Diff = || eps_theta(z_t, t) - eps ||_2^2
                    loss = F.mse_loss(noise_pred.float(), noise.float())

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                epoch_loss += loss.item()
                progress_bar.set_postfix({"loss": epoch_loss / (step + 1)})

            epoch_losses.append(epoch_loss / (step + 1))

            if (epoch + 1) % DIFFUSION_VAL_INTERVAL == 0:
                unet.eval()
                val_loss = 0.0

                with torch.no_grad():
                    for val_step, raw_batch_data in enumerate(valid_loader, start=1):
                        batch_data: BatchData = raw_batch_data
                        images = batch_data["image"].to(device)

                        with autocast(device.type, enabled=amp_enabled):
                            z_mu, z_sigma = autoencoder.encode(images)
                            z = autoencoder.sampling(z_mu, z_sigma)

                            noise = torch.randn_like(z)
                            timesteps = torch.randint(
                                0,
                                inferer.scheduler.num_train_timesteps,
                                (z.shape[0],),
                                device=z.device,
                            ).long()

                            noise_pred = inferer(
                                inputs=images,
                                diffusion_model=unet,
                                noise=noise,
                                timesteps=timesteps,
                                autoencoder_model=autoencoder,
                            )

                            loss = F.mse_loss(noise_pred.float(), noise.float())

                        val_loss += loss.item()

                val_loss /= val_step
                val_losses.append(val_loss)
                print(f"Diffusion epoch {epoch + 1} val loss: {val_loss:.4f}")

        return {"epoch_losses": epoch_losses, "val_losses": val_losses}


if __name__ == "__main__":
    print_library_versions()

    ensure_model_type_directories()
    run_dir = create_run_directory("LDM")
    plots_dir = run_dir / "plots"
    models_dir = run_dir / "models"
    plots_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    run_start_time = time.perf_counter()

    train_loader, valid_loader, _ = get_mednist_dataloaders()

    print(f"Training dataset size: {len(train_loader.dataset)}")
    print(f"Validation dataset size: {len(valid_loader.dataset)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = device.type == "cuda"

    autoencoder = LDMComponents.build_autoencoder(device)
    summary_kwargs = dict(col_names=["input_size", "output_size", "num_params"], depth=3, verbose=0)
    summary(autoencoder, (1, 1, IMAGE_SIZE, IMAGE_SIZE), device="cpu", **summary_kwargs)
    autoencoder.to(device)

    perceptual_loss_fn = build_alex_perceptual_loss(device)
    adversarial_loss_fn = PatchAdversarialLoss(criterion="least_squares")
    discriminator = LDMComponents.build_discriminator(device)

    optimizer_g = torch.optim.Adam(autoencoder.parameters(), lr=AUTOENCODER_LEARNING_RATE)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=DISCRIMINATOR_LEARNING_RATE)
    scaler_g = GradScaler(device.type, enabled=amp_enabled)
    scaler_d = GradScaler(device.type, enabled=amp_enabled)

    hyperparameters = {
        "model_type": "LDM",
        "common": {
            "batch_size": BATCH_SIZE,
            "image_size": IMAGE_SIZE,
            "train_valid_ratio": TRAIN_VALID_RATIO,
            "selected_label": SELECTED_LABEL,
        },
        "ldm_specific": {
            "autoencoder_channels": AUTOENCODER_CHANNELS,
            "latent_channels": LATENT_CHANNELS,
            "num_res_blocks": NUM_RES_BLOCKS,
            "norm_num_groups": NORM_NUM_GROUPS,
            "attention_levels": ATTENTION_LEVELS,
            "with_encoder_nonlocal_attn": WITH_ENCODER_NONLOCAL_ATTN,
            "with_decoder_nonlocal_attn": WITH_DECODER_NONLOCAL_ATTN,
            "discriminator_num_layers_d": DISCRIMINATOR_NUM_LAYERS_D,
            "discriminator_channels": DISCRIMINATOR_CHANNELS,
            "autoencoder_learning_rate": AUTOENCODER_LEARNING_RATE,
            "discriminator_learning_rate": DISCRIMINATOR_LEARNING_RATE,
            "diffusion_learning_rate": DIFFUSION_LEARNING_RATE,
            "kl_weight": KL_WEIGHT,
            "perceptual_weight": PERCEPTUAL_WEIGHT,
            "adversarial_weight": ADVERSARIAL_WEIGHT,
            "autoencoder_max_epochs": AUTOENCODER_MAX_EPOCHS,
            "autoencoder_val_interval": AUTOENCODER_VAL_INTERVAL,
            "autoencoder_warm_up_n_epochs": AUTOENCODER_WARM_UP_N_EPOCHS,
            "diffusion_max_epochs": DIFFUSION_MAX_EPOCHS,
            "diffusion_val_interval": DIFFUSION_VAL_INTERVAL,
            "diffusion_channels": DIFFUSION_CHANNELS,
            "diffusion_attention_levels": DIFFUSION_ATTENTION_LEVELS,
            "diffusion_num_head_channels": DIFFUSION_NUM_HEAD_CHANNELS,
            "diffusion_norm_num_groups": DIFFUSION_NORM_NUM_GROUPS,
            "diffusion_num_train_timesteps": DIFFUSION_NUM_TRAIN_TIMESTEPS,
            "diffusion_beta_start": DIFFUSION_BETA_START,
            "diffusion_beta_end": DIFFUSION_BETA_END,
            "diffusion_schedule": DIFFUSION_SCHEDULE,
            "diffusion_num_inference_steps": DIFFUSION_NUM_INFERENCE_STEPS,
        },
    }
    save_json(hyperparameters, run_dir / "hyperparameters.json")

    autoencoder_metrics = LDMTraining.train_autoencoder(
        autoencoder=autoencoder,
        discriminator=discriminator,
        perceptual_loss_fn=perceptual_loss_fn,
        adversarial_loss_fn=adversarial_loss_fn,
        train_loader=train_loader,
        valid_loader=valid_loader,
        optimizer_g=optimizer_g,
        optimizer_d=optimizer_d,
        scaler_g=scaler_g,
        scaler_d=scaler_d,
        device=device,
    )

    torch.save(autoencoder.state_dict(), models_dir / "autoencoderkl_for_diffusion_state_dict.pth")

    del discriminator
    del perceptual_loss_fn
    if device.type == "cuda":
        torch.cuda.empty_cache()

    unet = LDMComponents.build_diffusion_unet(device)
    LDMComponents.summarize_unet(unet)
    unet.to(device)

    scheduler = LDMComponents.build_scheduler()
    scale_factor = LDMTraining.compute_scale_factor(autoencoder, train_loader, device)
    inferer = LatentDiffusionInferer(scheduler, scale_factor=scale_factor)

    optimizer_unet = torch.optim.Adam(unet.parameters(), lr=DIFFUSION_LEARNING_RATE)
    scaler_unet = GradScaler(device.type, enabled=amp_enabled)

    diffusion_metrics = LDMTraining.train_diffusion(
        autoencoder=autoencoder,
        unet=unet,
        inferer=inferer,
        train_loader=train_loader,
        valid_loader=valid_loader,
        optimizer=optimizer_unet,
        scaler=scaler_unet,
        device=device,
    )

    torch.save(unet.state_dict(), models_dir / "diffusion_model_unet_state_dict.pth")

    metrics = {
        "scale_factor": scale_factor,
        "autoencoder": autoencoder_metrics,
        "diffusion": diffusion_metrics,
        "run_duration_seconds": time.perf_counter() - run_start_time,
    }
    save_json(metrics, run_dir / "metrics.json")
