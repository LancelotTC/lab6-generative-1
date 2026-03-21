# This contains imports and the loading of the data necessary in all three approaches
from common import *

# Parameters
spatial_dims = 2
in_channels = 1
out_channels = 1
channels = (16, 32, 64)
latent_channels = 3
num_res_blocks = 2
norm_num_groups = channels[0]
attention_levels = (False, False, False)

##
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# use of the with command line to avoid the automatic display of log information during the instanciation of the class model
with contextlib.redirect_stdout(None):
    model = AutoencoderKL(
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=channels,
        latent_channels=latent_channels,
        num_res_blocks=num_res_blocks,
        norm_num_groups=norm_num_groups,
        attention_levels=attention_levels,
    )
    model.to(device)

### Discriminator

# Parameters
spatial_dims = 2
in_channels = 1
out_channels = 1
num_layers_d = 3
channels = 16

# use of the with command line to avoid the automatic display of log information during the instanciation of the class model
with contextlib.redirect_stdout(None):
    discriminator = PatchDiscriminator(
        spatial_dims=spatial_dims,
        num_layers_d=num_layers_d,
        channels=channels,
        in_channels=in_channels,
        out_channels=out_channels,
        activation=(Act.LEAKYRELU, {"negative_slope": 0.2}),
        norm="BATCH",
        bias=False,
        padding=1,
    )
    discriminator.to(device)

# Print the summary of the network
summary_kwargs = dict(col_names=["input_size", "output_size", "num_params"], depth=3, verbose=0)
summary(discriminator, (1, 1, image_size, image_size), device="cpu", **summary_kwargs)


## Specify loss and optimization functions
learning_rate_g = 1e-4
learning_rate_d = 5e-4

p_loss = PerceptualLoss(spatial_dims=2, network_type="alex")
p_loss.to(device)
adv_loss = PatchAdversarialLoss(criterion="least_squares")
p_loss.to(device)
l1_loss = nn.L1Loss()
l1_loss.to(device)


def vae_gaussian_kl_loss(mu, sigma):
    kl_loss = 0.5 * torch.sum(mu.pow(2) + sigma.pow(2) - torch.log(sigma.pow(2)) - 1, dim=[1, 2, 3])
    kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
    return kl_loss


def reconstruction_loss(x_reconstructed, x):
    return l1_loss(x_reconstructed.float(), x.float())


def perceptual_loss(x_reconstructed, x):
    return p_loss(x_reconstructed.float(), x.float())


def loss_function(recon_x, x, mu, sigma, kl_weight, p_weight, a_weight, logits, target_is_real, for_discriminator):
    recon_loss = reconstruction_loss(recon_x, x)
    kld_loss = vae_gaussian_kl_loss(mu, sigma)
    p_loss = perceptual_loss(recon_x, x)
    a_loss = adv_loss(logits, target_is_real=target_is_real, for_discriminator=for_discriminator)
    return (
        recon_loss + kl_weight * kld_loss + p_weight * p_loss + a_weight * a_loss,
        recon_loss.item(),
        kl_weight * kld_loss.item(),
        p_weight * p_loss.item(),
        a_weight * a_loss.item(),
    )


# Specify optimizers
optimizer_generator = torch.optim.Adam(model.parameters(), lr=learning_rate_g)
optimizer_discriminator = torch.optim.Adam(params=discriminator.parameters(), lr=learning_rate_d)


### Training of the adversarial network


def train():
    # Number of epochs to train the model
    n_epochs = 120
    kl_weight = 1e-6  # KL divergence weight loss  / default monai value: 1e-6
    perceptual_weight = 1e-3  # Perceptual weight for loss / default monai value: 1e-3
    adversarial_weight = 1e-2  # Adversarial weight for generator loss / default monai value 1e-2

    # Move the model to the device
    model.to(device)
    discriminator.to(device)

    # Lists to store loss and accuracy for each epoch
    train_generator_loss_list = []
    train_discriminator_loss_list = []
    valid_metric_list = []
    reconstruction_metric_list = []
    kld_metric_list = []
    perceptual_metric_list = []
    adversarial_metric_list = []

    save_best_model_from_metric = True
    best_valid_metric = float("inf")  # to track the best validation mesasure
    best_model = None  # to store the best model
    best_epoch = 0  # to track the epoch number of the best model

    model.train()  # prepare model for training

    for epoch in range(n_epochs):
        # monitor training loss
        model.train()  # ensure the model is in training mode
        discriminator.train()
        train_generator_loss = 0
        train_discriminator_loss = 0
        reconstruction_metric = 0
        kld_metric = 0
        perceptual_metric = 0
        adversarial_metric = 0

        ################################
        # train the adversarial models #
        ################################
        for batch_data in train_loader:

            ###################
            # Generator part
            ###################

            # Load data and target samples stored the current batch_data
            inputs = batch_data["image"].to(device)
            # clear the gradients of all optimized variables
            optimizer_generator.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            reconstruction, z_mu, z_sigma = model(inputs)
            # predict fake logits
            logits_fake = discriminator(reconstruction.contiguous().float())[-1]
            # calculate the generator loss
            loss_generator, reconstruction_val, kld_val, perceptual_val, adversarial_val = loss_function(
                reconstruction,
                inputs,
                z_mu,
                z_sigma,
                kl_weight,
                perceptual_weight,
                adversarial_weight,
                logits_fake,
                target_is_real=True,
                for_discriminator=False,
            )

            # backpropagate
            loss_generator.backward()
            # perform a single optimization step (parameter update)
            optimizer_generator.step()
            # update running training generator loss
            train_generator_loss += loss_generator.item() * inputs.size(0)
            reconstruction_metric += reconstruction_val * inputs.size(0)
            kld_metric += kld_val * inputs.size(0)
            perceptual_metric += perceptual_val * inputs.size(0)
            adversarial_metric += adversarial_val * inputs.size(0)

            ###################
            # Discriminator part
            ###################

            if adversarial_weight > 0:

                # clear the gradients of all optimized variables
                optimizer_discriminator.zero_grad(set_to_none=True)
                # predict fake logits
                logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
                # compute real logits
                logits_real = discriminator(inputs.contiguous().detach())[-1]
                # calculate the discriminator loss
                loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
                loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
                discriminator_loss = (loss_d_fake + loss_d_real) * 0.5
                loss_discriminator = adversarial_weight * discriminator_loss

                # backpropagate
                loss_discriminator.backward()
                # perform a single optimization step (parameter update)
                optimizer_discriminator.step()
                # update running training discriminator loss
                train_discriminator_loss += loss_discriminator.item() * inputs.size(0)

        # Calculate average training loss and accuracy over the epoch
        train_generator_loss_list.append(train_generator_loss / len(train_loader.dataset))
        reconstruction_metric_list.append(reconstruction_metric / len(train_loader.dataset))
        kld_metric_list.append(kld_metric / len(train_loader.dataset))
        perceptual_metric_list.append(perceptual_metric / len(train_loader.dataset))
        adversarial_metric_list.append(adversarial_metric / len(train_loader.dataset))

        if adversarial_weight > 0:
            train_discriminator_loss_list.append(train_discriminator_loss / len(train_loader.dataset))

        ###################
        # Validation step #
        ###################
        model.eval()  # set model to evaluation mode
        valid_metric = 0

        with torch.no_grad():  # disable gradient calculation during validation
            for batch_data in valid_loader:
                # Load data and target samples stored the current batch_data
                inputs = batch_data["image"].to(device)
                # forward pass: compute predicted outputs by passing inputs to the model
                reconstruction, z_mu, z_sigma = model(inputs)
                # calculate the loss
                recon_val = reconstruction_loss(reconstruction.float(), inputs.float())
                valid_metric += recon_val.item() * inputs.size(0)

        # Compute average validation loss and accuracy
        valid_metric_list.append(valid_metric / len(valid_loader.dataset))

        print(
            f"Epoch: {epoch+1} \tTraining Loss: {train_generator_loss_list[-1]:.6f} \tValidation metric: {valid_metric_list[-1]:.6f}"
        )

        if save_best_model_from_metric:
            # Save the model if it has the best validation loss
            if valid_metric_list[-1] < best_valid_metric:
                best_valid_metric = valid_metric_list[-1]
                best_model = model.state_dict()
                best_epoch = epoch + 1  # Save the epoch number
        else:
            # Save the last model as best model
            best_valid_metric = valid_metric_list[-1]
            best_epoch = epoch + 1
            best_model = model.state_dict()

    # After training, load the best model
    model.load_state_dict(best_model)
    torch.save(best_model, "best_test_lossmodel.pth")  # Save the best model

    print(f"Best model selected at epoch {best_epoch} with validation loss: {best_valid_metric:.6f}")

    # Plot loss curves
    plt.figure(figsize=(16, 6))

    # Plotting global loss
    plt.subplot(1, 5, 1)
    plt.plot(train_generator_loss_list, color="C0", linewidth=2.0, label="Training Loss")
    plt.title("Training loss curve")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    # Plotting Reconstruction loss
    plt.subplot(1, 5, 2)
    plt.plot(reconstruction_metric_list, color="C0", linewidth=2.0, label="Reconstruction metric")
    plt.title("Reconstruction metric")
    plt.xlabel("Epochs")
    plt.ylabel("Loss (%)")
    plt.legend()

    # Plotting KL loss
    plt.subplot(1, 5, 3)
    plt.plot(kld_metric_list, color="C0", linewidth=2.0, label="KL divergence metric")
    plt.title("KL divergence metric")
    plt.xlabel("Epochs")
    plt.ylabel("Loss (%)")
    plt.legend()

    # Plotting Perceptual loss
    plt.subplot(1, 5, 4)
    plt.plot(perceptual_metric_list, color="C0", linewidth=2.0, label="Perceputal metric")
    plt.title("Perceptual metric")
    plt.xlabel("Epochs")
    plt.ylabel("Loss (%)")
    plt.legend()

    # Plotting Perceptual loss
    plt.subplot(1, 5, 5)
    plt.plot(adversarial_metric_list, color="C0", linewidth=2.0, label="Adversarial metric")
    plt.title("Adversarial metric")
    plt.xlabel("Epochs")
    plt.ylabel("Loss (%)")
    plt.legend()

    if adversarial_weight > 0:
        # Express the adversarial terms without the weight coefficient
        generator_list = [x / adversarial_weight for x in adversarial_metric_list]
        discriminator_list = [x / adversarial_weight for x in train_discriminator_loss_list]

        plt.title("Adversarial Training Curves", fontsize=20)
        plt.plot(np.linspace(1, n_epochs, n_epochs), generator_list, color="C0", linewidth=2.0, label="Generator")
        plt.plot(
            np.linspace(1, n_epochs, n_epochs), discriminator_list, color="C1", linewidth=2.0, label="Discriminator"
        )
        plt.yticks(fontsize=12)
        plt.xticks(fontsize=12)
        plt.xlabel("Epochs", fontsize=16)
        plt.ylabel("Loss", fontsize=16)
        plt.legend(prop={"size": 14})

        # Save the figure as a PNG file
        plt.savefig("img/adversarial_training_curves.png", dpi=300, bbox_inches="tight")

        plt.show()

    # initialize lists to monitor test loss and accuracy
    test_metric = 0.0

    model.eval()  # prep model for *evaluation*

    with torch.no_grad():  # disable gradient calculation during validation
        for batch_data in test_loader:
            # forward pass: compute predicted outputs by passing inputs to the model
            inputs = batch_data["image"].to(device)
            reconstruction, z_mu, z_sigma = model(inputs)
            # calculate the loss
            recon_val = reconstruction_loss(reconstruction.float(), inputs.float())
            test_metric += recon_val.item() * inputs.size(0)

    # calculate and print avg test loss
    test_metric = test_metric / len(test_loader.dataset)
    print("Test reconstruction metric: {:.6f}\n".format(test_metric))

    # Prepare next cell
    dataiter = iter(test_loader)

    # obtain one batch of test images
    batch_data = next(dataiter)
    batch_data = next(dataiter)

    # get sample outputs
    inputs = batch_data["image"].to(device)
    recons, _, _ = model(inputs)
    # reconstruction images for display
    recons = recons.detach().cpu().numpy()
    inputs = inputs.detach().cpu().numpy()

    # Plot the image, label and prediction
    fig = plt.figure(figsize=(8, 8))
    for idx in range(3):
        ax = fig.add_subplot(3, 2, 2 * idx + 1, xticks=[], yticks=[])
        ax.imshow(np.squeeze(inputs[idx]), cmap="gray")
        ax.set_title("Original")
        ax = fig.add_subplot(3, 2, 2 * idx + 2, xticks=[], yticks=[])
        ax.imshow(np.squeeze(recons[idx]), cmap="gray")
        ax.set_title("Reconstructed")

    downsampling_ratio = 1
    counter = 0  # counter initialisation

    model.eval()  # prep model for *evaluation*
    z_mu_accumulated = []

    with torch.no_grad():  # Deactivate the gradient computations
        for batch_data in test_loader:
            counter += 1
            if counter % downsampling_ratio == 0:
                # forward pass: compute predicted outputs by passing inputs to the model
                inputs = batch_data["image"].to(device)
                z_mu, _ = model.encode(inputs)
                z_mu_accumulated.append(z_mu.cpu().numpy())

    z_mu_accumulated = np.concatenate(z_mu_accumulated, axis=0)

    z_mu_flattened = z_mu_accumulated.reshape(z_mu_accumulated.shape[0], -1)
    print(
        f"Size of the latent matrix passed to the t-SNE method (Nb Sample, vector dimensionality) = {z_mu_flattened.shape}"
    )

    # Apply t-SNE to reduce the dimensionality to 2 and allows a visualization of the latent space
    tsne = TSNE(n_components=2, random_state=42)
    z_mu_tsne = tsne.fit_transform(z_mu_flattened)

    random_generation = False

    if random_generation:
        z_mu = torch.randn(1, 4, 8, 8).to(device)
    else:
        inputs = batch_data["image"].to(device)
        print(inputs.shape)
        input = inputs[2]
        input = input.unsqueeze(0)
        print(f"The input image is of size {input.shape}")
        z_mu, _ = model.encode(input)

    print(f"The latent sample is of size {z_mu.shape}")

    # Decode latent sample
    reconstruction = model.decode(z_mu)
    print(f"The reconstrudted image is of size {reconstruction.shape}")

    img = reconstruction.squeeze().detach().cpu().numpy()
    img = np.squeeze(img)

    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, xticks=[], yticks=[])
    ax.imshow(img, cmap="gray")

    # Randomly select two points in the latent space
    inputs = batch_data["image"].to(device)
    input = inputs[2]
    input = input.unsqueeze(0)
    latent_1, _ = model.encode(input)
    input = inputs[4]
    input = input.unsqueeze(0)
    latent_2, _ = model.encode(input)

    synthetic_1 = model.decode(latent_1)
    synthetic_2 = model.decode(latent_2)

    img_1 = synthetic_1.squeeze().detach().cpu().numpy()
    img_1 = np.squeeze(img_1)
    img_2 = synthetic_2.squeeze().detach().cpu().numpy()
    img_2 = np.squeeze(img_2)

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(1, 2, 1, xticks=[], yticks=[])
    ax.imshow(img_1, cmap="gray")
    ax = fig.add_subplot(1, 2, 2, xticks=[], yticks=[])
    ax.imshow(img_2, cmap="gray")


    # Interpolate between the two points and decode to generate images
    images = interpolate_images(model, latent_1, latent_2, steps=64)

    # Animate the interpolated images
    filename = "mnist_interpolation.gif"
    save_animation_as_gif(images, filename=filename, interval=100)

    # Affiche le GIF dans Jupyter
    display(Image(filename=filename))


def interpolate_images(model, latent_1, latent_2, steps=10):
    # Interpolate between point1 and point2 in the latent space

    latent_1.to(device)
    latent_2.to(device)
    t_values = torch.linspace(0, 1, steps).to(device)
    latent_tmp = [torch.lerp(latent_1, latent_2, t).to(device) for t in t_values]
    latent_interp = torch.stack([latent.squeeze(0) for latent in latent_tmp], dim=0)
    synthetic_interp = model.decode(latent_interp)

    # Return images as a list after detaching and converting to numpy
    return [img.squeeze().detach().cpu().numpy() for img in synthetic_interp]


def save_animation_as_gif(images, filename="animation.gif", interval=200):
    fig, ax = plt.subplots(figsize=(2, 2))
    img_display = ax.imshow(images[0], cmap="gray", vmin=0, vmax=1)
    ax.axis("off")

    def update(frame):
        img_display.set_data(images[frame])
        return [img_display]

    ani = FuncAnimation(fig, update, frames=len(images), interval=interval, blit=True)
    ani.save(filename, writer="pillow", fps=1000 // interval)
    plt.close(fig)


if __name__ == "__main__":
    train()
