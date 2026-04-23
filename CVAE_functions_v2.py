# ======================================
# CVAE function library
# ======================================

import torch # PyTorch library
import torch.nn as nn # Module containing neural network layers (Conv, Linear, etc.)
import numpy as np # NumPy for array operations
import matplotlib.pyplot as plt
from PIL import Image
from pandas import *
import easygui

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

# =====================
# AJOUT des ATTRIBUTES
# =====================

def weight(latent_vectors, attributes, data_directions, data_vectors):
    """
    Modify latent vectors based on a list of attributes.

    The function loads attribute directions and mean vectors from '.npz' files,
    then applies transformations in latent space. The attribute "Bald" is treated
    as a special case using a mean vector instead of a direction.

    Parameters
    ----------
    latent_vectors : numpy.ndarray
        Array of latent vectors with shape (batch_size, latent_dim).
    attributes : list of str
        List containing only the attributes to apply out of ["Smiling", "Mustache", "Eyeglasses", "Bald"].
        If no attribute, list empty. If only smiling, attributes = ["Smiling"].
    data_directions : numpy.lib.npyio.NpzFile
        Loaded '.npz' file containing attribute direction vectors
    data_vectors : numpy.lib.npyio.NpzFile
        Loaded '.npz' file containing attribute mean vectors

    Returns
    -------
    z_modified : numpy.ndarray
        Modified latent vectors with the same shape as 'latent_vectors'.
    """

    # Convert NpzFile objects into Python dictionaries
    attribute_directions = {k: data_directions[k] for k in data_directions.keys()}
    attribute_vectors = {k: data_vectors[k] for k in data_vectors.keys()}
    # Each key corresponds to an attribute name (e.g., "Smiling")

    # Copy to avoid overwriting original latent vectors
    z_modified = latent_vectors.copy()

    for attr in attributes:

        # Special case: using mean vector instead of direction for bald for better results
        if attr == "Bald":
            z_modified += 7 * attribute_vectors["Bald"]
            # Less robust than directions but sometimes visually more effective

        # Apply other attributes directions (stronger, more meaningful edits)
        else:
            z_modified += 10 * attribute_directions[attr]
            # Scale factor (10) controls intensity of modification

    return z_modified


# ==========================
# FUSION + MUTATION
# ==========================
def fuse_parents(parent_matrix,weight):
    """
    weightd merge all vectors together

    Parameters
    ----------
    parent_matrix : numpy.array
        matrice of all vectors, number of row correspond to the number of vectors to fuse

    weight : numpy.array or list
        vector containing the weights of each vector in the same order as the matrice

    Returns
    ----------
    new_image : numpy.array
        New vector computed by the weighted merge

    Examples
    ----------
    >>>fuse_parents(np.array([[0.5,0.8,0.2,0.6,0.7],[0.4,0.7,0.1,0.5,0.8]]), (1,2))
    [0.43333333 0.73333333 0.13333333 0.53333333 0.76666667]
    """
    new_image=[]
    div = sum(weight)
    for col in range(0,np.shape(parent_matrix)[1]):
        for vec in range(0,np.shape(parent_matrix)[0]):
            parent_matrix[vec,col] = parent_matrix[vec,col]*weight[vec]/div
    new_image=np.sum(matrice,axis=0)
    return new_image


def mutate_vector(vector, num_variants, mutation_rate, mutation_level):
    """
    produce multuple variations (mutations) from an initial parent vector

    Parameters
    ----------
    vector : numpy.array
        Initial vector to produce variations from

    num_variants : int
        Number of variations to produce

    mutation_rate : int or float
        Amount of mutations to apply

    mutation_level : int or float
        Intensity of mutations to apply

    Returns
    -------
    np.array(mutated_vectors)
        vectors produced by the random mutations

    Examples
    --------
    >>> mutate_vector(np.array([0.4,0.3,0.9,0.8,0.2,0.1,0.4]), num_variants=3, mutation_rate=0.4, mutation_level=0.1)
    [[0.42754856 0.26299389 0.9        0.8        0.2        0.1       0.3276892 ]
    [0.41722165 0.3460928  0.85578323 0.8        0.19935116 0.19189771     0.4       ]
    [0.4        0.3        0.9507284  0.8        0.2        0.15455378     0.27444405]]
    """

    mutated_vectors=[]
    for i in range (num_variants):
        vect_mutation=vector.copy()
        mute=np.random.rand(len(vector)) #random var between 0 and 1, values are kept if they are inferior to the mutation rate
        noise=np.random.normal(0,mutation_level, len(vector))#var representing the intensity of the mutations between 0 and 1
        for i in range(len(vector)):
            if mute[i]<mutation_rate:
                vect_mutation[i]+=noise[i] #add the mutation to the corressponding value
            else:
                i=i+1
        mutated_vectors.append(vect_mutation)#create a list of mutated vectors
    return np.array(mutated_vectors)

def image():
    """
    Browse through your files to find the image you wish to import

    Parameters
    ----------
    none

    Returns
    ----------
    The chosen image
    """

    myimg = Image.open(easygui.fileopenbox()).convert("RGB")
    #myimg.show()
    img=np.array(myimg)
    return img

def grey_scale_img(img):
    """
    Transform a colored image into a grayscale one and resize into size 64x64

    Parameters
    ----------
    img : numpy.array
        The image you want to transform

    Returns
    ----------
    reduced_img : numpy.array
        The grayscale and resized image

    """

    temp_img = img.copy()
    temp_img = img[:,:,0]*0.2126+img[:,:,1]*0.7152+img[:,:,2]*0.0722
    temp_img = temp_img.astype(np.uint8)
    plt.imshow(temp_img,cmap='gray', vmin=0, vmax=255) ;
    plt.show()
    if img.shape[:2] != (64,64):
        image_pillow = Image.fromarray(temp_img)
        new_size = (64, 64) # 2. On définie la nouvelle taille
        # 3. Redimensionnement avec filtre de haute qualité Lanczos
        reduced_img = image_pillow.resize(new_size, Image.Resampling.LANCZOS)

    else:
        print("the size of the image is right")
        reduced_img=temp_img

    plt.imshow(reduced_img,cmap='gray', vmin=0, vmax=255)
    plt.show()
    # Save the image
    #reduced_img.save("image_64x64.jpg", quality=95)
    return(reduced_img)
