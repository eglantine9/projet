# This file contains scripts used to generate the pth and npz files. 

# ======================================
# CVAE ARCHITECTURE
# ======================================

import torch # PyTorch library
import torch.nn as nn # Module containing neural network layers (Conv, Linear, etc.)

class VAE(nn.Module):
    """
    Variational Autoencoder (VAE) with convolutional encoder and decoder.
    This model encodes grayscale images into a latent space and reconstructs
    them using a transposed convolutional decoder.

    Parameters
    ----------
    latent_dim : int
        Dimensionality of the latent space.

    Attributes
    ----------
    latent_dim : int
        Size of the latent representation.

    encoder_conv : nn.Sequential
        Convolutional encoder network.

    flatten : nn.Flatten
        Layer used to flatten convolutional feature maps.

    fc_mu : nn.Linear
        Fully connected layer producing the latent mean.

    fc_logvar : nn.Linear
        Fully connected layer producing the latent log-variance.

    decoder_fc : nn.Linear
        Fully connected layer mapping latent vectors to feature maps.

    decoder : nn.Sequential
        Transposed convolutional decoder network.
    """
    def __init__(self, latent_dim):
        super().__init__() # Initialize parent nn.Module

        self.latent_dim = latent_dim # Latent space size

        # ======================
        # Convolutional encoder
        # ======================
        self.encoder_conv = nn.Sequential( # Stacks multiple layers automatically
            nn.Conv2d(1, 16, 3, padding=1), # Convolution 1->16, keeps size (64x64): 1 input channel (grayscale), 16 filters, kernel of size 3, 1 pixel of padding, stride 1
            nn.LayerNorm([16, 64, 64]), # Layer normalization
            nn.ReLU(), # Rectified Linear Unit activation, introduces non-linearity
            nn.MaxPool2d(2), # Downsampling 64->32,

            nn.Conv2d(16, 32, 3, padding=1), # Convolution 16->32, size 32x32
            nn.LayerNorm([32, 32, 32]),
            nn.ReLU(),
            nn.MaxPool2d(2), # Downsampling 32->16

            nn.Conv2d(32, 64, 3, padding=1), # Convolution 32->64, size 16x16
            nn.LayerNorm([64, 16, 16]),
            nn.ReLU(),
            nn.MaxPool2d(2), # Downsampling 16->8
        )

        self.flatten = nn.Flatten() # Flatten (64,8,8) -> vector of size 64*8*8

        # ======================
        # Latent space
        # ======================
        self.fc_mu = nn.Linear(64 * 8 * 8, latent_dim) # Project vector to mu
        self.fc_logvar = nn.Linear(64 * 8 * 8, latent_dim) # Project vector to logvar

        # ======================
        # Convolutional decoder
        # ======================
        self.decoder_fc = nn.Linear(latent_dim, 64 * 8 * 8) # Latent -> feature map 64*8*8

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), # Upsample 8->16, 64 in channels, 32 out channels, kernel of size 3, 2 pixels at a time, (out)padding 1,
            nn.LayerNorm([32, 16, 16]), # Layer normalization
            nn.ReLU(),

            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), # Upsample 16->32
            nn.LayerNorm([16, 32, 32]),
            nn.ReLU(),

            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1), # Upsample 32->64
            nn.Sigmoid() # Output pixels between 0 and 1
        )

    def encode(self, x):
        """
        Encode input images into latent distribution parameters.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, 1, 64, 64).

        Returns
        -------
        mu : torch.Tensor
            Mean of the latent distribution, shape (batch_size, latent_dim).

        logvar : torch.Tensor
            Log-variance of the latent distribution, shape (batch_size, latent_dim).
        """
        x = self.encoder_conv(x) # Pass image through convolutional encoder
        x = self.flatten(x) # Flatten for fully connected layer
        mu = self.fc_mu(x) # Compute latent mean
        logvar = self.fc_logvar(x) # Compute latent log-variance
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """
        Apply the reparameterization trick to sample from the latent space.

        Parameters
        ----------
        mu : torch.Tensor
            Mean of the latent distribution.

        logvar : torch.Tensor
            Log-variance of the latent distribution.

        Returns
        -------
        z : torch.Tensor
            Sampled latent vector.
        """
        std = torch.exp(0.5 * logvar) # Standard deviation
        eps = torch.randn_like(std) # Random noise
        return mu + eps * std # z = mu + sigma * epsilon

    def decode(self, z):
        """
        Decode latent vectors into reconstructed images.

        Parameters
        ----------
        z : torch.Tensor
            Latent tensor of shape (batch_size, latent_dim).

        Returns
        -------
        x_recon : torch.Tensor
            Reconstructed images of shape (batch_size, 1, 64, 64).
        """
        x = self.decoder_fc(z) # Latent -> feature map
        x = x.view(-1, 64, 8, 8) # Reshape for convolutional decoder, (batchsize, channels, height, width) for feature map
        return self.decoder(x) # Final reconstructed image

    def forward(self, x):
        """
        Forward pass through the VAE.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, 1, 64, 64).

        Returns
        -------
        x_recon : torch.Tensor
            Reconstructed images.

        mu : torch.Tensor
            Mean of the latent distribution.

        logvar : torch.Tensor
            Log-variance of the latent distribution.
        """
        mu, logvar = self.encode(x) # Encode the image
        z = self.reparameterize(mu, logvar) # Sample latent vector
        x_recon = self.decode(z) # Reconstruct image
        return x_recon, mu, logvar  # Return reconstructed image, mu, logvar


# =====================================
# VAE MODEL LOADING + ENCODE + DECODE
# =====================================

import torch # PyTorch library for tensor operations and deep learning
import numpy as np # NumPy for array operations
from CVAE_architecture import VAE # Import the VAE architecture

# Use GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# LOAD TRAINED VAE MODEL
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



# ENCODE IMAGE -> LATENT VECTOR
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

# DECODE LATENT VECTOR -> IMAGE
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

# ======================================
# CVAE TRAIN, generation of pth file 
# ======================================

import torch # PyTorch library
import torch.nn.functional as F # Functional API for loss functions, activations, etc.
from torch.utils.data import DataLoader, Dataset, Subset # Dataset and DataLoader utilities
from torchvision import datasets, transforms # Standard datasets and image transformations
from CVAE_architecture import VAE # VAE architecture


# PARAMETERS

LATENT_DIM = 512 # Dimension of the latent space
BATCH_SIZE = 64 # Mini-batch size for training
EPOCHS = 10 # Number of training epochs
MODEL_PATH = "vae_celeba_final.pth" # Path to save the trained model
BETA = 1e-3  # Weight for KL divergence in VAE loss

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# CELEBA DATASET

transform = transforms.Compose([
    transforms.CenterCrop(178), # Crop central 178x178 region
    transforms.Resize(64), # Resize to 64x64
    transforms.Grayscale(), # Convert to grayscale
    transforms.ToTensor() # Convert Python Imaging Library image to PyTorch tensor
])

celeba_dataset = datasets.CelebA(
    root="./data", # Path where images and attributes are saved
    split="train", # loader uses training set (not "valid" or "test" subsets)
    target_type="attr", # Attributes are kept but ignored for unsupervised training
    transform=transform, # Applying transformations beyond
    download=False # Data already downloaded
)


# DATASET (IGNORE LABELS)

class CelebADataset(Dataset):
    """
    Dataset wrapper to return only images (ignore attributes/labels).

    Parameters
    ----------
    dataset : torchvision.datasets.CelebA
        Original CelebA dataset containing images and attributes.
    """
    def __init__(self, dataset):
        self.dataset = dataset # Store original dataset

    def __len__(self):
        return len(self.dataset) # Number of samples

    def __getitem__(self, idx):
        x, _ = self.dataset[idx] # Get image, ignore label
        return x

# Wrap dataset
train_dataset = CelebADataset(celeba_dataset)


# DATA LOADER

loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True, # Shuffle data each epoch for randomisation
    num_workers=0 # No parallel loading because training realized under windows
)


# MODEL AND OPTIMIZER
model = VAE(LATENT_DIM).to(device) # Initialize VAE with latent dimension
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) # Adam optimizer, learning rate controls how big each update step is

# LOSS FUNCTION
def loss_function(x_recon, x, mu, logvar):
    """
    Compute VAE loss (reconstruction + Kullback-Leibler divergence).
    KL divergence measures how one probability distrobution differs from another.

    Parameters
    ----------
    x_recon : torch.Tensor
        Reconstructed images from decoder (batch_size, 1, 64, 64).
    x : torch.Tensor
        Original input images (batch_size, 1, 64, 64).
    mu : torch.Tensor
        Latent mean (batch_size, latent_dim).
    logvar : torch.Tensor
        Latent log-variance (batch_size, latent_dim).

    Returns
    -------
    torch.Tensor
        Scalar loss value.
    """
    recon_loss = F.binary_cross_entropy(x_recon, x, reduction='mean') # Reconstruction loss
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp()) # KL divergence
    return recon_loss + BETA * kl_loss # Weighted sum


# TRAINING LOOP
for epoch in range(EPOCHS):
    model.train() # Set model to training mode
    running_loss = 0 # Track epoch loss

    for x in loader: # Iterate over mini-batches
        x = x.to(device) # Move batch to GPU/CPU

        optimizer.zero_grad() # Clear previous gradients

        # All 64 images are processed in parallel, and tensors are given as outputs containing all reconstructed images in the batch
        x_recon, mu, logvar = model(x) # Forward pass

        loss = loss_function(x_recon, x, mu, logvar) # Compute VAE loss, one scaalr loss for the whole batch

        loss.backward() # Backpropagate gradients, compute gradients with respect to all learnable parameters
        optimizer.step() # Update weights

        running_loss += loss.item() # Accumulate batch loss

    # Print average loss for the epoch
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {running_loss/len(loader):.4f}") # Divide loss by number of batches in epoch

    # Save intermediate checkpoint each epoch to compare how many epochs are needed
    torch.save({
        "model_state_dict": model.state_dict(),
        "latent_dim": LATENT_DIM
    }, f"vae_celeba{epoch+1}.pth")

    print("VAE checkpoint saved successfully.")


# FINAL MODEL SAVE
torch.save({
    "model_state_dict": model.state_dict(),
    "latent_dim": LATENT_DIM
}, MODEL_PATH)

print("VAE saved successfully.")

# =======================================
# VAE ATTRIBUTES MEAN VECTORS, generation of npz file 
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



# PARAMETERS
ATTRIBUTES = ["Eyeglasses", "Smiling", "Bald", "Mustache"]

# Select GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# LOAD MODEL

# Load trained VAE and its latent dimension
model, latent_dim = load_model("vae_celeba_final.pth")


# DATASET

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


# COMPUTE MEAN VECTORS

attribute_vectors = compute_attribute_vectors(model, celeba_dataset, ATTRIBUTES, batch_size=64)


# SAVE TO FILE

# Save all attribute vectors into a .npz file
np.savez("attribute_vectors.npz", **attribute_vectors) # Unpack dictionary
print("Attribute vectors saved to attribute_vectors.npz")

# =======================================
# VAE ATTRIBUTES DISTRIBUTION, generation of npz file 
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



# PARAMETERS
ATTRIBUTES = ["Eyeglasses", "Smiling", "Bald", "Mustache"]

# Use GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# LOAD MODEL

# Load trained VAE and latent dimension
model, latent_dim = load_model("vae_celeba_final.pth")


# DATASET

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



# COMPUTE DIRECTIONS

attribute_directions = compute_attribute_directions(model, celeba_dataset, ATTRIBUTES, batch_size=64)


# SAVE

# Save all directions into a .npz file
np.savez("attribute_directions.npz", **attribute_directions)
print("Attribute directions saved to attribute_directions.npz")


