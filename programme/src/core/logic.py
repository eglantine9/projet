import torch
import numpy as np
from .CVAE_architecture import VAE as AutoEncoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_trained_model(path):
    """
    Load a trained AutoEncoder model from a .pth file.
    """
    try:
        # Map location to device (cpu/cuda)
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        latent_dim = checkpoint["latent_dim"]

        model = AutoEncoder(latent_dim).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        return model, latent_dim
    except Exception as e:
        print(f"Error loading model from {path}: {e}")
        raise

def encode_images(model, images_tensor):
    """
    Encode image tensors into latent vectors (NumPy).
    """
    with torch.no_grad():
        images_tensor = images_tensor.to(device)
        mu, logvar = model.encode(images_tensor)
        # Use the mean of the latent distribution as the latent vector
        z = mu
    return z.cpu().numpy()

def decode_latents(model, latent_numpy):
    """
    Decode latent vectors (NumPy) back into image tensors.
    """
    with torch.no_grad():
        latent_numpy = np.atleast_2d(latent_numpy)
        z = torch.tensor(latent_numpy, dtype=torch.float32).to(device)
        x = model.decode(z)
    return x.cpu()

from .algorithme_genetique import fusion as fuse_parents
from .algorithme_genetique import mutation as mutate_vector


def fuse_weighted(latents, weights):
    """
    Weighted average of latent vectors with norm-preservation.
    """
    weights = np.array(weights, dtype=np.float64)
    if weights.sum() == 0:
        weights = np.ones_like(weights)
    weights = weights / weights.sum()
    
    # Simple linear combination pulls towards the blurry origin center
    fused_linear = np.average(latents, weights=weights, axis=0)
    
    # Calculate the targeted norm (weighted average of parent norms)
    norms = np.linalg.norm(latents, axis=1)
    target_norm = np.average(norms, weights=weights)
    
    # Scale fused_linear exactly to target hypersphere
    current_norm = np.linalg.norm(fused_linear)
    if current_norm > 1e-6:
        return fused_linear * (target_norm / current_norm)
    return fused_linear


def load_attribute_directions(path):
    """
    Load attribute direction vectors from a .npz file.
    """
    data = np.load(path)
    return {key: data[key] for key in data.files}
