######################################################
# CHARGEMENT MODELE + FONCTIONS ENCODE/DECODE
# Output: images <-> vecteur numpy
######################################################

"""
Utility functions for loading a trained AutoEncoder and converting
between image space and latent space.

This module provides helper functions to:

- Load a trained AutoEncoder model
- Encode an image into its latent representation
- Decode a latent vector into a reconstructed image

These utilities simplify the interaction between PyTorch tensors and
NumPy arrays, which we need for the genetic algorithm. 

Notes
-----
All computations automatically run on GPU if available, otherwise on CPU.

Expected tensor shapes:

- Image tensor: (batch_size, 1, 64, 64)
- Latent vector: (batch_size, latent_dim)

Examples
--------
Load a trained model:

>>> model, latent_dim = load_model("autoencoder_olivetti.pth")

Encode an image:

>>> z = encode_image(model, image_tensor)

Decode a latent vector:

>>> image = decode_latent(model, z)
"""

import torch
import numpy as np
from FINAL_model_architecture import AutoEncoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(path):
    """
    Load a trained AutoEncoder model from a checkpoint file.

    Parameters
    ----------
    path : str
        Path to the saved model checkpoint (.pth file).

    Returns
    -------
    model : AutoEncoder
        Loaded AutoEncoder model in evaluation mode.

    latent_dim : int
        Dimension of the latent space used during training.

    Notes
    -----
    The checkpoint must contain:

    - "model_state_dict" : trained weights
    - "latent_dim" : latent space dimension
    """
    checkpoint = torch.load(path, map_location=device)
    latent_dim = checkpoint["latent_dim"]

    model = AutoEncoder(latent_dim).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, latent_dim


def encode_image(model, image_tensor):
    """
    Encode an image tensor into its latent representation.

    Parameters
    ----------
    model : AutoEncoder
        Trained autoencoder model.

    image_tensor : torch.Tensor
        Input tensor representing the image.
        Expected shape: (batch_size, 1, 64, 64).

    Returns
    -------
    numpy.ndarray
        Latent representation of the image with shape
        (batch_size, latent_dim) = (number of samples, size latent vector for each sample)
        2-dimensional NumPy array where each row corresponds to one encoded sample
        and each column corresponds to one latent featur. 

    Notes
    -----
    Gradient computation is disabled during encoding.We don't want to modify the weights anymore.
    """
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        z = model.encode(image_tensor)
    return z.cpu().numpy()


def decode_latent(model, latent_numpy):
    """
    Decode a latent vector into an image reconstruction.

    Parameters
    ----------
    model : AutoEncoder
        Trained autoencoder model.

    latent_numpy : numpy.ndarray
        Latent vector or batch of latent vectors with shape
        (batch_size, latent_dim).

    Returns
    -------
    torch.Tensor
        Reconstructed image tensor with shape
        (batch_size, 1, 64, 64).

    Notes
    -----
    The latent vector is converted to a PyTorch tensor before being
    passed through the decoder network.
    """
    with torch.no_grad():
        z = torch.tensor(latent_numpy, dtype=torch.float32).to(device)
        x = model.decode(z)
    return x.cpu()
