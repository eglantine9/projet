# =======================================
# VAE ATTRIBUTES MEAN VECTORS
# USED FOR "BALD" ATTRIBUTE
# =======================================

import torch # PyTorch library for tensors and deep learning
import numpy as np # NumPy for numerical operations
from CVAE_latent import load_model, encode_image # Custom functions to load model and encode images
from torchvision import datasets, transforms # Dataset and preprocessing tools
from torch.utils.data import Subset # Utility to create dataset subsets
import random # For random sampling

def compute_attribute_vectors(model, dataset, attr_names, batch_size=64):
    """
    Compute mean latent vectors for images with each attribute.
    Simple average over latent vectors of images having the attribute.

    Parameters
    ----------
    model : VAE
        Pretrained VAE model.
    dataset : torchvision.datasets.CelebA
        CelebA dataset.
    attr_names : list of str
        List of attribute names to compute vectors for.
    batch_size : int
        Batch size used for encoding images.

    Returns
    -------
    dict
        Dictionary of {attribute_name: mean_latent_vector (np.array)}.
    """

    # Get index of each attribute in CelebA attribute list
    attr_indices = [dataset.attr_names.index(a) for a in attr_names]

    attr_vectors = {} # Dictionary to store mean vectors

    # Loop over each attribute
    for attr, idx in zip(attr_names, attr_indices):
        print(f"Computing mean vector for: {attr}")

        # Select indices of images that have this attribute (value == 1)
        indices = [i for i in range(len(dataset)) if dataset[i][1][idx] == 1]

        latent_list = [] # Store latent vectors for this attribute

        # Process images in batches to avoid memory issues
        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i:i+batch_size] # Select batch indices

            # Extract images from dataset (ignore attributes)
            # dataset: [(img1,attr1), (img2,attr2), ...]
            batch_imgs, _ = zip(*[dataset[j] for j in batch_idx]) # Result: (img1,img2,...), (attr1, attr2,...)

            # Stack list of tensors into a single tensor (batch_size, 1, 64, 64)
            batch_imgs = torch.stack(batch_imgs).to(device)

            # Encode images into latent vectors
            batch_latent = encode_image(model, batch_imgs)

            # Store latent vectors
            latent_list.append(batch_latent)

        # Stack all batches into a single array (num_samples, latent_dim)
        latent_vectors = np.vstack(latent_list)

        # Compute mean latent vector for this attribute
        mean_vector = np.mean(latent_vectors, axis=0)

        # Store result in dictionary
        attr_vectors[attr] = mean_vector

        print(f"{attr} -> mean latent shape: {mean_vector.shape}")

    return attr_vectors


# ======================
# PARAMETERS
# ======================
ATTRIBUTES = ["Eyeglasses", "Smiling", "Bald", "Mustache"]

# Select GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================
# LOAD MODEL
# ======================
# Load trained VAE and its latent dimension
model, latent_dim = load_model("vae_celeba_final.pth")

# ======================
# DATASET
# ======================
# Define preprocessing pipeline
transform = transforms.Compose([
    transforms.CenterCrop(178), # Crop central part of image
    transforms.Resize(64), # Resize to 64x64
    transforms.Grayscale(), # Convert to grayscale (1 channel)
    transforms.ToTensor() # Convert to tensor [0,1]
])

# Load CelebA dataset
celeba_dataset = datasets.CelebA(
    root="./data",
    split="train", # Use training split
    target_type="attr", # Load attributes
    transform=transform,
    download=False
)

# ======================
# COMPUTE MEAN VECTORS
# ======================
attribute_vectors = compute_attribute_vectors(model, celeba_dataset, ATTRIBUTES, batch_size=64)

# ======================
# SAVE TO FILE
# ======================
# Save all attribute vectors into a .npz file
np.savez("attribute_vectors.npz", **attribute_vectors) # Unpack dictionary
print("Attribute vectors saved to attribute_vectors.npz")
