import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from keras.datasets.mnist import load_data

# -----------------------------------------------------------------------------
# Data Loading and Preprocessing
# -----------------------------------------------------------------------------
# Load and normalize the MNIST dataset
(trainX, trainy), (testX, testy) = load_data()
trainX = trainX.astype(np.float32) / 255.0  # Normalize to [0,1]

# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------
def sample_latent(batch_size: int, device: torch.device) -> torch.Tensor:
    """
    Sample random latent vectors from a uniform distribution.
    """
    return torch.rand((batch_size, 50), device=device)


def get_minibatch(batch_size: int, device: torch.device) -> torch.Tensor:
    """
    Randomly select a minibatch of images from the training dataset.
    Reshape images into vectors.
    """
    indices = torch.randperm(trainX.shape[0])[:batch_size]
    batch = torch.tensor(trainX[indices], dtype=torch.float).reshape(batch_size, -1)
    return batch.to(device)


# -----------------------------------------------------------------------------
# Network Definitions
# -----------------------------------------------------------------------------
class Generator(nn.Module):
    """
    Generator: Maps a latent vector (50-D) to a flattened MNIST image (784-D).
    """
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(50, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024, eps=1e-5, momentum=0.1, affine=False),
            nn.ReLU(),
            nn.Linear(1024, 784),
            nn.Sigmoid()  # Ensures output is in [0,1]
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.model(z)


class Encoder(nn.Module):
    """
    Encoder: Maps a flattened MNIST image (784-D) to a latent vector (50-D).
    """
    def __init__(self):
        super(Encoder, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024, eps=1e-5, momentum=0.1, affine=False),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 50),
            nn.Sigmoid()  # Produces latent code in [0,1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class Discriminator(nn.Module):
    """
    Discriminator: Distinguishes between joint pairs (image, latent code).
    Expects concatenated input of an image (784-D) and a latent vector (50-D).
    """
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(784 + 50, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024, eps=1e-5, momentum=0.1, affine=False),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 1),
            nn.Sigmoid()  # Outputs a probability
        )

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        # Concatenate latent vector and image
        xz = torch.cat((z, x), dim=1)
        return self.model(xz)


# -----------------------------------------------------------------------------
# Training Function
# -----------------------------------------------------------------------------
def train_model(
    discriminator: nn.Module,
    generator: nn.Module,
    encoder: nn.Module,
    optimizers: list,
    schedulers: list,
    batch_size: int = 128,
    device: torch.device = torch.device("cpu"),
    nb_epochs: int = 187500,
) -> None:
    """
    Trains the adversarial feature learning network.
    """
    bce_loss = nn.BCELoss()

    for epoch in tqdm(range(nb_epochs), desc="Training"):
        # Sample latent noise and real images
        z_sample = sample_latent(batch_size, device)
        x_real = get_minibatch(batch_size, device)

        # Generate fake images from latent codes
        x_generated = generator(z_sample)
        # Encode real images into latent codes
        z_encoded = encoder(x_real)

        # Discriminator predictions
        # For generated pairs, we want D to output 1 (for generator/encoder objective)
        d_pred_generated = discriminator(x_generated, z_sample).view(-1)
        # For encoded pairs, we want D to output 0 (for generator/encoder objective)
        d_pred_encoded = discriminator(x_real, z_encoded).view(-1)

        # Compute binary cross entropy losses
        loss_generated = bce_loss(d_pred_generated, torch.ones(batch_size, device=device))
        loss_encoded = bce_loss(d_pred_encoded, torch.zeros(batch_size, device=device))
        adv_loss = loss_generated + loss_encoded

        # Zero gradients for all optimizers
        for opt in optimizers:
            opt.zero_grad()

        # Backpropagation
        adv_loss.backward()

        # Update parameters
        for opt in optimizers:
            opt.step()

        # Update learning rate schedulers in the latter half of training
        if epoch > (nb_epochs / 2):
            for scheduler in schedulers:
                scheduler.step()


# -----------------------------------------------------------------------------
# Main Execution Block
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Select device (GPU if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Instantiate networks
    generator = Generator().to(device)
    encoder = Encoder().to(device)
    discriminator = Discriminator().to(device)

    # Setup optimizers.
    # Note: The 'maximize=True' flag is used for generator and encoder so that they
    # perform gradient ascent (i.e., try to fool the discriminator).
    optimizer_G = optim.Adam(
        generator.parameters(), lr=2e-4, betas=(0.5, 0.999), weight_decay=2.5e-5, maximize=True
    )
    optimizer_E = optim.Adam(
        encoder.parameters(), lr=2e-4, betas=(0.5, 0.999), weight_decay=2.5e-5, maximize=True
    )
    optimizer_D = optim.Adam(
        discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999), weight_decay=2.5e-5, maximize=False
    )
    optimizers = [optimizer_G, optimizer_E, optimizer_D]

    # Learning rate schedulers for each optimizer
    schedulers = [
        optim.lr_scheduler.ExponentialLR(opt, gamma=0.9999508793911394) for opt in optimizers
    ]

    # Train the networks
    train_model(discriminator, generator, encoder, optimizers, schedulers, device=device)

    # -----------------------------------------------------------------------------
    # Visualization
    # -----------------------------------------------------------------------------
    with torch.no_grad():
        # Generate images from random latent vectors
        z_vis = sample_latent(20, device)
        generated_images = generator(z_vis)
        # Get a minibatch of real images and reconstruct via encoder->generator
        real_images = get_minibatch(20, device)
        z_encoded = encoder(real_images)
        reconstructed_images = generator(z_encoded)

    # Plot the images: first row = G(z), second row = real x, third row = G(E(x))
    plt.figure(figsize=(18, 3.5))
    for i in range(20):
        # Row 1: Generated images from latent vectors
        plt.subplot(3, 20, 1 + i)
        plt.axis('off')
        if i == 0:
            plt.title('G(z)', fontsize=17)
        plt.imshow(generated_images[i].cpu().numpy().reshape(28, 28), cmap='gray')

        # Row 2: Real images
        plt.subplot(3, 20, 21 + i)
        plt.axis('off')
        if i == 0:
            plt.title('x', fontsize=17)
        plt.imshow(real_images[i].cpu().numpy().reshape(28, 28), cmap='gray')

        # Row 3: Reconstructed images from real images via encoder and generator
        plt.subplot(3, 20, 41 + i)
        plt.axis('off')
        if i == 0:
            plt.title('G(E(x))', fontsize=17)
        plt.imshow(reconstructed_images[i].cpu().numpy().reshape(28, 28), cmap='gray')

    plt.savefig("Imgs/adversarial_feature_learning.png")
    plt.show()
