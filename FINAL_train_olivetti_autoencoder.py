######################################################
# ENTRAINEMENT + SAUVEGARDE OLIVETTI FACES AUTOENCODER 
# Output: sauvegarde du modèle entraîné
######################################################

# train_olivetti_autoencoder.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import fetch_olivetti_faces
from FINAL_model_architecture import AutoEncoder

# ======================
# Paramètres
# ======================

LATENT_DIM = 32
BATCH_SIZE = 32
EPOCHS = 50
MODEL_PATH = "autoencoder_olivetti.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================
# Dataset
# ======================

data = fetch_olivetti_faces(shuffle=True, random_state=42)
images = torch.tensor(data.images, dtype=torch.float32).unsqueeze(1)

dataset = TensorDataset(images, images)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ======================
# Modèle
# ======================

model = AutoEncoder(LATENT_DIM).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# ======================
# Entraînement
# ======================

for epoch in range(EPOCHS):

    model.train()
    running_loss = 0

    for x, _ in loader:

        x = x.to(device)

        optimizer.zero_grad()
        x_recon, _ = model(x)
        loss = criterion(x_recon, x)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {running_loss/len(loader):.6f}")

# ======================
# Sauvegarde
# ======================

torch.save({
    "model_state_dict": model.state_dict(),
    "latent_dim": LATENT_DIM
}, MODEL_PATH)

print("Modèle sauvegardé.")