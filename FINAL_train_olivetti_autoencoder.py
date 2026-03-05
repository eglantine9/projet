######################################################
# TEST PIPELINE OLIVETTI AUTOENCODER
# Output: encodage + reconstruction + visualisation
######################################################

"""
Test the full AutoEncoder pipeline on the Olivetti Faces dataset.

This script demonstrates how to use a trained AutoEncoder model to:

1. Load the trained model from disk
2. Encode face images into latent vectors
3. Decode latent vectors into reconstructed images
4. Visualize the original and reconstructed images

The goal is to verify that the trained model correctly compresses and
reconstructs face images using its latent representation.

Notes
-----
Dataset characteristics:

- Dataset: Olivetti Faces
- Number of images: 400
- Image size: 64 x 64
- Channels: 1 (grayscale)

Pipeline steps:

1. Load trained model checkpoint
2. Select a batch of test images
3. Encode images into latent vectors and display 
4. Decode latent vectors into reconstructed images
5. Display original vs reconstructed images

Tensor shapes involved:

- Input images: (batch_size, 1, 64, 64)
- Latent vectors: (batch_size, latent_dim)
- Reconstructed images: (batch_size, 1, 64, 64)

Examples
--------
Run the test script:

>>> python3 test_olivetti_pipeline.py

Expected output:

- Printed latent vector shape, type and constitution 
- Visualization comparing original and reconstructed faces
"""

# test_olivetti_pipeline.py

import torch
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from FINAL_latent_interface_olivetti import load_model, encode_image, decode_latent


# ======================
# Charger modèle
# ======================

model, latent_dim = load_model("autoencoder_olivetti.pth")
print("Latent dim =", latent_dim)


# ======================
# Charger images
# ======================

data = fetch_olivetti_faces(shuffle=True, random_state=42)  # 42 pour fixer le générateur aléatoire
images = torch.tensor(data.images, dtype=torch.float32).unsqueeze(1)

sample_imgs = images[8:16]


# ======================
# ENCODE
# ======================

latent_vectors = encode_image(model, sample_imgs)
print("Shape latent:", latent_vectors.shape)
print(type(latent_vectors))
print(latent_vectors)


# ======================
# DECODE
# ======================

reconstructed = decode_latent(model, latent_vectors)
reconstructed = reconstructed.squeeze(1)


# ======================
# VISUALISATION
# ======================

fig, axes = plt.subplots(2, 8, figsize=(16, 4))

for i in range(8):
    axes[0, i].imshow(sample_imgs[i].squeeze(0), cmap='gray')
    axes[0, i].axis('off')
    axes[1, i].imshow(reconstructed[i], cmap='gray')
    axes[1, i].axis('off')

axes[0, 0].set_title("Original")
axes[1, 0].set_title("Decoded")

plt.show()
