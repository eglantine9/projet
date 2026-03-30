####################################################
# ARCHITECTURE finale de l'AutoEncoder
# Output: class utilisable dans les autres scripts
####################################################

import torch
import torch.nn as nn


class AutoEncoder(nn.Module):
    """
    Convolutional AutoEncoder for image compression and reconstruction.

    This model encodes a grayscale image into a low-dimensional latent
    representation and then reconstructs the image from this latent space.
    The encoder consists of convolutional layers that progressively reduce
    the spatial resolution, while the decoder uses transposed convolutions
    to reconstruct the original image.

    Parameters
    ----------
    latent_dim : int
        Dimension of the latent space representation.

    Attributes
    ----------
    encoder : torch.nn.Sequential
        Convolutional encoder network that maps the input image to the latent vector.

    decoder_fc : torch.nn.Linear
        Fully connected layer that reshapes the latent vector into a tensor
        suitable for the convolutional decoder.

    decoder : torch.nn.Sequential
        Convolutional decoder network that reconstructs the image from the
        intermediate tensor representation.

    Notes
    -----
    Expected input image size: (1, 64, 64)

    The encoder reduces the spatial dimensions as follows:

    - 64x64 -> 32x32
    - 32x32 -> 16x16
    - 16x16 -> 8x8

    The decoder performs the inverse operation using transposed convolutions.
    """

    def __init__(self, latent_dim):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, latent_dim)
        )

        self.decoder_fc = nn.Linear(latent_dim, 64 * 8 * 8)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def encode(self, x):
        """
        Encode the input image into a latent representation.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, 1, 64, 64).

        Returns
        -------
        torch.Tensor
            Latent representation of shape (batch_size, latent_dim).
        """
        return self.encoder(x)

    def decode(self, z):
        """
        Decode a latent vector into a reconstructed image.

        Parameters
        ----------
        z : torch.Tensor
            Latent tensor of shape (batch_size, latent_dim).

        Returns
        -------
        torch.Tensor
            Reconstructed image tensor of shape (batch_size, 1, 64, 64).
        """
        x = self.decoder_fc(z)
        x = x.view(-1, 64, 8, 8)
        return self.decoder(x)

    def forward(self, x):
        """
        Perform a full forward pass through the autoencoder.

        The input image is encoded into the latent space and then decoded
        back into a reconstructed image.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, 1, 64, 64).

        Returns
        -------
        tuple of torch.Tensor
            x_recon : torch.Tensor
                Reconstructed image tensor of shape (batch_size, 1, 64, 64).

            z : torch.Tensor
                Latent representation of shape (batch_size, latent_dim).
        """
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z
