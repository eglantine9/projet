import torch
import numpy as np
from .models import AutoEncoder

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

def fuse_parents(parent_matrix):
    """
    Calculate the average of multiple parent latent vectors.
    """
    # Mean across columns (features) for each sample if matrix, or just mean of samples
    return np.mean(parent_matrix, axis=0)

def mutate_vector(vector, num_variants, mutation_rate, mutation_level):
    """
    Create mutated variations of a latent vector.
    """
    mutated_vectors = []
    vector = vector.squeeze()
    
    for _ in range(num_variants):
        variant = vector.copy()
        # Random mask for which genes mutate
        mask = np.random.rand(len(vector))
        # Gaussian noise for the mutation
        noise = np.random.normal(0, mutation_level, len(vector))
        # Apply mutation where mask < rate
        variant[mask < mutation_rate] += noise[mask < mutation_rate]
        mutated_vectors.append(variant)
        
    return np.array(mutated_vectors)
