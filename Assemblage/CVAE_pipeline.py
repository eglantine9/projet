# ===================================================
# PIPELINE VAE CELEBA
# Output: encoding + fusion + mutation + attributes modification + reconstruction + visualization
# ===================================================

import torch # PyTorch for tensor operations
import numpy as np # NumPy for numerical operations
import matplotlib.pyplot as plt # Visualization
from torchvision import datasets, transforms # Dataset + preprocessing
from CVAE_functions import mutation, fusion, load_model, encode_image, decode_latent # Custom VAE utilities

# ======================
# PARAMETERS
# ======================
BATCH_SIZE = 8 # Number of images processed at once

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================
# LOAD VAE MODEL
# ======================
# Load trained model and latent dimension
model, latent_dim = load_model("vae_celeba_final.pth")
print("Latent dim:", latent_dim)

# ======================
# LOAD ATTRIBUTE DATA
# ======================
# Load precomputed attribute directions and mean vectors
data_directions = np.load("attribute_directions.npz")
data_vectors = np.load("attribute_vectors.npz")

# Convert NpzFile objects into Python dictionaries
attribute_directions = {k: data_directions[k] for k in data_directions.keys()}
attribute_vectors = {k: data_vectors[k] for k in data_vectors.keys()}
# Each key corresponds to an attribute name (e.g., "Smiling")

# ======================
# LOAD CELEBA TEST DATASET
# ======================
# Define preprocessing pipeline
transform = transforms.Compose([
    transforms.CenterCrop(178), # Crop center region
    transforms.Resize(64), # Resize to 64x64
    transforms.Grayscale(), # Convert to grayscale
    transforms.ToTensor() # Convert PIL image to tensor
])

# Load test split of CelebA
celeba_dataset = datasets.CelebA(
    root="./data",
    split="test", # Use test set, non-trained images
    target_type="attr", # Load attributes (not used here)
    transform=transform,
    download=False
)

# ======================
# SAMPLE A BATCH OF IMAGES
# ======================
# Take first BATCH_SIZE images
indices = torch.arange(BATCH_SIZE)

# Extract images (ignore attributes)
sample_imgs, _ = zip(*[celeba_dataset[i] for i in indices])

# Stack into tensor of shape (BATCH_SIZE, 1, 64, 64)
sample_imgs = torch.stack(sample_imgs)

# ======================
# ENCODE IMAGES -> LATENT SPACE
# ======================
# Convert images into latent vectors
latent_vectors = encode_image(model, sample_imgs)
print("Shape latent:", latent_vectors.shape)
# Expected: (BATCH_SIZE, latent_dim)

# ======================
# FUSION
# ======================
# Fsuion of botained latent_vectors
vecteur_fusion = fusion(latent_vectors)
print("Fusion vector:", vecteur_fusion)

# ======================
# MUTATION
# ======================
# Mutation of obtained fusion vector
vecteurs_mutants = mutation(
    vecteur_fusion,
    nb_variants=8,
    taux_mutation=0.2,
    niveau_mutation=0.5
)
print("Mutants shape:", vecteurs_mutants.shape)
print("Mutated vectors:", vecteurs_mutants)

# ======================
# MODIFY LATENT VECTORS
# ======================
Bald = False # Decide if we want bald attribute or not
ATTRIBUTES = ["Eyeglasses"] # Put in this list the wanted attributes out of: ["Eyeglasses", "Smiling", "Mustache"]

# Copy vectors that will be modified (important to not overwrite originals)
z_modified = vecteurs_mutants.copy()

# Apply attribute directions
for attr in ATTRIBUTES:
    z_modified += 10 * attribute_directions[attr]
    # Scale factor (10) controls intensity of modification

# Special case: using mean vector instead of direction for bald for better results
if Bald:
    z_modified += 7 * attribute_vectors["Bald"]
    # Less robust than directions but sometimes visually more effective

# ======================
# DECODE LATENT -> IMAGE SPACE
# ======================
# Decode fusioned, mutated, modified latent vectors
reconstructed_modified = decode_latent(model, z_modified).squeeze(1)
# Decode original latent vectors
reconstructed_original = decode_latent(model, latent_vectors).squeeze(1)

# ======================
# VISUALIZATION
# ======================
# Create grid: 3 rows (original / recon / modified), figsize=(width=2 inches per image, height=6 inches)
fig, axes = plt.subplots(3, BATCH_SIZE, figsize=(BATCH_SIZE * 2, 6))

for i in range(BATCH_SIZE):
    # Row 1: original images
    axes[0, i].imshow(sample_imgs[i].squeeze(0), cmap='gray')
    axes[0, i].axis('off')

    # Row 2: reconstructed images
    axes[1, i].imshow(reconstructed_original[i], cmap='gray')
    axes[1, i].axis('off')

    # Row 3: modified images
    axes[2, i].imshow(reconstructed_modified[i], cmap='gray')
    axes[2, i].axis('off')


# Add row labels on the left
fig.text(0.04, 0.98, "Original", ha='center', rotation='horizontal')
fig.text(0.04, 0.65, "Reconstruction", ha='center', rotation='horizontal')
fig.text(0.04, 0.33, "Modified", ha='center', rotation='horizontal')

# Adjust layout to avoid overlaps
plt.tight_layout()
# Display figure
plt.show()
