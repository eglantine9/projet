######################################################
# CHARGEMENT MODELE + FONCTIONS ENCODE/DECODE
# Output: images <-> vecteur numpy
######################################################

# latent_interface_olivetti.py
import torch
import numpy as np
from FINAL_model_architecture import AutoEncoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(path):
    checkpoint = torch.load(path, map_location=device)
    latent_dim = checkpoint["latent_dim"]

    model = AutoEncoder(latent_dim).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, latent_dim


def encode_image(model, image_tensor):
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        z = model.encode(image_tensor)
    return z.cpu().numpy()


def decode_latent(model, latent_numpy):
    with torch.no_grad():
        z = torch.tensor(latent_numpy, dtype=torch.float32).to(device)
        x = model.decode(z)
    return x.cpu()