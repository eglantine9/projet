# =======================================
# VAE ATTRIBUTES DISTRIBUTION
# USED FOR "SMILING", "EYEGLASSES", "MUSTACHE" ATTRIBUTES
# =======================================

import torch # PyTorch library for tensors and GPU computation
import numpy as np # NumPy for numerical operations
from CVAE_latent import load_model, encode_image # Custom functions for model loading and encoding
from torchvision import datasets, transforms # Dataset and preprocessing utilities
from torch.utils.data import Subset # Tool to create dataset subsets
import random # For random sampling


def compute_attribute_directions(model, dataset, attr_names, batch_size=64):
    """
    Compute latent attribute directions for each attribute.

    Each direction is defined as:
    direction = mean(latents with attribute) - mean(latents without attribute)

    The resulting direction is normalized to unit norm.

    Parameters
    ----------
    model : VAE
        Pretrained variational autoencoder.
    dataset : torchvision.datasets.CelebA
        CelebA dataset containing images and attribute labels.
    attr_names : list of str
        List of attribute names to compute directions for.
    batch_size : int, optional
        Number of images processed simultaneously during encoding.

    Returns
    -------
    dict
        Dictionary mapping each attribute name to a normalized direction vector
        (np.ndarray of shape (latent_dim,)).
    """
    # Get index of each attribute in CelebA attribute list
    attr_indices = [dataset.attr_names.index(a) for a in attr_names]

    directions = {} # Dictionary to store attribute directions

    # Loop over each attribute
    for attr, idx in zip(attr_names, attr_indices):
        print(f"Computing direction for: {attr}")

        # Select indices of images WITH the attribute (positive samples)
        pos_indices = [i for i in range(len(dataset)) if dataset[i][1][idx] == 1]

        # Select indices of images WITHOUT the attribute (negative samples)
        neg_indices = [i for i in range(len(dataset)) if dataset[i][1][idx] == 0]

        # ------------------------------
        # Helper function to compute mean latent vector
        # ------------------------------
        def compute_mean(indices):
            latent_list = [] # Store latent vectors for all batches

            # Process images in batches
            for i in range(0, len(indices), batch_size):
                batch_idx = indices[i:i+batch_size] # Select batch indices

                # Extract images (ignore attributes)
                batch_imgs, _ = zip(*[dataset[j] for j in batch_idx])

                # Stack into tensor of shape (batch_size, 1, 64, 64)
                batch_imgs = torch.stack(batch_imgs).to(device)

                # Encode images into latent space
                batch_latent = encode_image(model, batch_imgs)

                # Store batch latent vectors
                latent_list.append(batch_latent)

            # Stack all batches into one array (num_samples, latent_dim)
            latent_vectors = np.vstack(latent_list)

            # Compute mean latent vector column-wise
            return np.mean(latent_vectors, axis=0)

        # Compute mean latent vectors for both groups
        mean_pos = compute_mean(pos_indices) # With attribute
        mean_neg = compute_mean(neg_indices) # Without attribute

        # Compute direction: difference between the two means
        direction = mean_pos - mean_neg

        # Normalize direction to unit length
        direction /= np.linalg.norm(direction)

        # Store result
        directions[attr] = direction

        print(f"{attr} direction computed. Norm: {np.linalg.norm(direction):.3f}")

    return directions


# ======================
# PARAMETERS
# ======================
ATTRIBUTES = ["Eyeglasses", "Smiling", "Bald", "Mustache"]

# Use GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================
# LOAD MODEL
# ======================
# Load trained VAE and latent dimension
model, latent_dim = load_model("vae_celeba_final.pth")

# ======================
# DATASET
# ======================
# Define preprocessing pipeline
transform = transforms.Compose([
    transforms.CenterCrop(178), # Crop center of image
    transforms.Resize(64), # Resize to 64x64
    transforms.Grayscale(), # Convert to grayscale
    transforms.ToTensor() # Convert to tensor in [0,1]
])

# Load CelebA dataset
celeba_dataset = datasets.CelebA(
    root="./data",
    split="train", # Use training set
    target_type="attr", # Load attribute labels
    transform=transform,
    download=False
)


# ======================
# COMPUTE DIRECTIONS
# ======================
attribute_directions = compute_attribute_directions(model, celeba_dataset, ATTRIBUTES, batch_size=64)

# ======================
# SAVE
# ======================
# Save all directions into a .npz file
np.savez("attribute_directions.npz", **attribute_directions)
print("Attribute directions saved to attribute_directions.npz")
