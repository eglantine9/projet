# ======================================
# CVAE function library
# ======================================

import torch # PyTorch library
import torch.nn as nn # Module containing neural network layers (Conv, Linear, etc.)
import numpy as np # NumPy for array operations


# Use GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


# ======================
# LOAD TRAINED VAE MODEL
# ======================
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


# ======================
# ENCODE IMAGE -> LATENT VECTOR
# ======================
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


# ======================
# DECODE LATENT VECTOR -> IMAGE
# ======================
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


# ==========================
# FUSION + MUTATION (old version, has to be reformed)
# ==========================

def fusion(matrice):
    # moyenne des vecteurs parents
    new_vector = np.mean(matrice, axis=0)
    return new_vector
 #fais la moyenne des colonnes de la matrice
#On fait la moyene de chaque colonnes de la matrice pour ne former qu'un seul vecteur

def mutation(vecteur, nb_variants, taux_mutation, niveau_mutation):
    vecteurs_mutes=[]
    for i in range (nb_variants):
        vect_mutation=vecteur.copy()
        mute=np.random.rand(len(vecteur)) #variable aléatoire entre 0 et 1, elles representes les valeurs qui seront mutés si < taux de mutation
        bruit=np.random.normal(0,niveau_mutation, len(vecteur))#petite variable representant la mutation entre 0 et niveau de mutation
        vect_mutation[mute < taux_mutation] += bruit[mute < taux_mutation] #on ajoute la mutation a la valeur de l'indice où survient une mutation
        vecteurs_mutes.append(vect_mutation)#On crée une liste de vecteur mutés
    return np.array(vecteurs_mutes)
#Mute est une liste de meme taille que le vecteur, les valeurs de chacunes de ses cases seront générés aléatoirement et comprises entre 0 et 1
#bruit est un vecteur ou toute les valeurs sont des valeurs de mutations( proche de 0 car suit une loi normale)
#Pour les valeurs de Mute qui seront inférieur au taux de mutation, on garde l'indice, cet indice sera celui qui dans le vect_mutation permettra de muté une valeur précise avec l'ajout du bruit[i]
