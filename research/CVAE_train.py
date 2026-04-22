import torch # PyTorch library
import torch.nn.functional as F # Functional API for loss functions, activations, etc.
from torch.utils.data import DataLoader, Dataset, Subset # Dataset and DataLoader utilities
from torchvision import datasets, transforms # Standard datasets and image transformations
from CVAE_architecture import VAE # VAE architecture

# ======================
# PARAMETERS
# ======================
LATENT_DIM = 512 # Dimension of the latent space
BATCH_SIZE = 64 # Mini-batch size for training
EPOCHS = 10 # Number of training epochs
MODEL_PATH = "vae_celeba_final.pth" # Path to save the trained model
BETA = 1e-3  # Weight for KL divergence in VAE loss

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================
# CELEBA DATASET
# ======================
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

# ======================
# DATASET (IGNORE LABELS)
# ======================
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

# ======================
# DATA LOADER
# ======================
loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True, # Shuffle data each epoch for randomisation
    num_workers=0 # No parallel loading because training realized under windows
)

# ======================
# SUBSET FOR DEBUG
# ======================
# subset_size = 10000
# indices = torch.randperm(len(train_dataset))[:subset_size]
# subset_dataset = Subset(train_dataset, indices)
#
# loader = DataLoader(
#     subset_dataset,
#     batch_size=BATCH_SIZE,
#     shuffle=True,
#     num_workers=0
# )

# ======================
# MODEL AND OPTIMIZER
# ======================
model = VAE(LATENT_DIM).to(device) # Initialize VAE with latent dimension
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) # Adam optimizer, learning rate controls how big each update step is

# ======================
# LOSS FUNCTION
# ======================
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

# ======================
# TRAINING LOOP
# ======================
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

# ======================
# FINAL MODEL SAVE
# ======================
torch.save({
    "model_state_dict": model.state_dict(),
    "latent_dim": LATENT_DIM
}, MODEL_PATH)

print("VAE saved successfully.")
