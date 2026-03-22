from GAN import GANComponents, summary
from LDM import LDMComponents
from pathlib import Path

model_structures_folder = Path("model_structures")
model_structures_folder.mkdir(exist_ok=True)

summary_kwargs = dict(col_names=["input_size", "output_size", "num_params"], depth=3, verbose=0)

# GAN
print("--- VAE Generator ---")
with open(f"{model_structures_folder}/GAN - VAE Generator.txt", "w") as file:
    file.write(str(summary(GANComponents.build_generator("cuda"), (1, 1, 64, 64), **summary_kwargs)))  # VAE Generator

print("--- Discriminator ---")
with open(f"{model_structures_folder}/GAN - Discriminator.txt", "w") as file:
    file.write(
        str(summary(GANComponents.build_discriminator("cuda"), (1, 1, 64, 64), **summary_kwargs))
    )  # Discriminator


# LDM
print("--- VAE Generator ---")
with open(f"{model_structures_folder}/LDM - VAE Generator.txt", "w") as file:
    file.write(str(summary(LDMComponents.build_autoencoder("cuda"), (1, 1, 64, 64), **summary_kwargs)))  # VAE Generator

print("--- UNet Diffusion Model ---")
with open(f"{model_structures_folder}/LDM - UNet Diffusion Model.txt", "w") as file:
    file.write(str(LDMComponents.build_diffusion_unet("cuda")))  # UNet Diffusion Model

print("--- Discriminator ---")
with open(f"{model_structures_folder}/LDM - Discriminator.txt", "w") as file:
    file.write(
        str(summary(LDMComponents.build_discriminator("cuda"), (1, 1, 64, 64), **summary_kwargs))
    )  # Discriminator
