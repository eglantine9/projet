# =====================================
# VAE MODEL LOADING + ENCODE + DECODE
# =====================================

import torch # PyTorch library for tensor operations and deep learning
import numpy as np # NumPy for array operations
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "programme", "src", "core")))
from CVAE_architecture import VAE # Import the VAE architecture

# Use GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ======================
# LOAD TRAINED VAE MODEL
# ======================
def load_model(path):
    """
    Load a trained VAE model from a checkpoint file.

    Parameters
    ----------
    path : str
        Path to the saved PyTorch checkpoint (.pth).

    Returns
    -------
    model : VAE
        The loaded VAE model in evaluation mode.
    latent_dim : int
        Dimension of the latent space.
    """
    # Load checkpoint from disk, map to the correct device
    checkpoint = torch.load(path, map_location=device)

    # Extract latent space dimension from checkpoint
    latent_dim = checkpoint["latent_dim"]

    # Initialize the VAE model with the correct latent dimension
    model = VAE(latent_dim).to(device)

    # Load trained weights into the model
    model.load_state_dict(checkpoint["model_state_dict"])

    # Set the model to evaluation mode (disables dropout, batchnorm updates)
    model.eval()

    return model, latent_dim


# ======================
# ENCODE IMAGE -> LATENT VECTOR
# ======================
def encode_image(model, image_tensor):
    """
    Encode input images into latent vectors.

    Parameters
    ----------
    model : VAE
        The trained VAE model.
    image_tensor : torch.Tensor
        Input images of shape (batch_size, 1, 64, 64).

    Returns
    -------
    np.ndarray
        Latent vectors of shape (batch_size, latent_dim).
    """
    # Disable gradient computation for efficiency
    with torch.no_grad():
        # Move input tensor to the correct device
        image_tensor = image_tensor.to(device)

        # Encode images through the VAE encoder
        mu, logvar = model.encode(image_tensor)

        # Use the mean of the latent distribution as the latent vector
        z = mu

    # Convert tensor to NumPy array and move to CPU
    return z.cpu().numpy()


# ======================
# DECODE LATENT VECTOR -> IMAGE
# ======================
def decode_latent(model, latent_numpy):
    """
    Decode latent vectors back into reconstructed images.

    Parameters
    ----------
    model : VAE
        The trained VAE model.
    latent_numpy : np.ndarray
        Latent vectors of shape (latent_dim,) or (batch_size, latent_dim).

    Returns
    -------
    torch.Tensor
        Reconstructed images of shape (batch_size, 1, 64, 64) on CPU.
    """
    # Disable gradient computation for efficiency
    with torch.no_grad():
        # Ensure latent vector is at least 2D (batch dimension)
        latent_numpy = np.atleast_2d(latent_numpy)

        # Convert NumPy array to PyTorch tensor and move to device
        z = torch.tensor(latent_numpy, dtype=torch.float32).to(device)

        # Decode latent vector into reconstructed images
        x = model.decode(z)

    # Move reconstructed images to CPU and return
    return x.cpu()
