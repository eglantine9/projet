###############################################################################################################
# Script avec toutes les fonction nécessaires pour encoder, fusionner, muter, decoder les images sélectionnées
################################################################################################################

# importations nécessaires pour que cela fonctionne 
import torch
import torch.nn as nn
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

####################################################
# ARCHITECTURE finale de l'AutoEncoder
# Output: class utilisable dans les autres fonctions
####################################################
class AutoEncoder(nn.Module):
    """
    Convolutional AutoEncoder for image compression and reconstruction.

    This model encodes a grayscale image into a low-dimensional latent
    representation and then reconstructs the image from this latent space.
    The encoder consists of convolutional layers that progressively reduce
    the spatial resolution, while the decoder uses transposed convolutions
    to reconstruct the original image.

    Parameters
    ----------
    latent_dim : int
        Dimension of the latent space representation.

    Attributes
    ----------
    encoder : torch.nn.Sequential
        Convolutional encoder network that maps the input image to the latent vector.

    decoder_fc : torch.nn.Linear
        Fully connected layer that reshapes the latent vector into a tensor
        suitable for the convolutional decoder.

    decoder : torch.nn.Sequential
        Convolutional decoder network that reconstructs the image from the
        intermediate tensor representation.

    Notes
    -----
    Expected input image size: (1, 64, 64)

    The encoder reduces the spatial dimensions as follows:

    - 64x64 -> 32x32
    - 32x32 -> 16x16
    - 16x16 -> 8x8

    The decoder performs the inverse operation using transposed convolutions.
    """

    def __init__(self, latent_dim):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, latent_dim)
        )

        self.decoder_fc = nn.Linear(latent_dim, 64 * 8 * 8)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def encode(self, x):
        """
        Encode the input image into a latent representation.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, 1, 64, 64).

        Returns
        -------
        torch.Tensor
            Latent representation of shape (batch_size, latent_dim).
        """
        return self.encoder(x)

    def decode(self, z):
        """
        Decode a latent vector into a reconstructed image.

        Parameters
        ----------
        z : torch.Tensor
            Latent tensor of shape (batch_size, latent_dim).

        Returns
        -------
        torch.Tensor
            Reconstructed image tensor of shape (batch_size, 1, 64, 64).
        """
        x = self.decoder_fc(z)
        x = x.view(-1, 64, 8, 8)
        return self.decoder(x)

    def forward(self, x):
        """
        Perform a full forward pass through the autoencoder.

        The input image is encoded into the latent space and then decoded
        back into a reconstructed image.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, 1, 64, 64).

        Returns
        -------
        tuple of torch.Tensor
            x_recon : torch.Tensor
                Reconstructed image tensor of shape (batch_size, 1, 64, 64).

            z : torch.Tensor
                Latent representation of shape (batch_size, latent_dim).
        """
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z


######################################################
# CHARGEMENT MODELE + FONCTIONS ENCODE/DECODE
# Output: images <-> vecteur numpy
######################################################

"""
Utility functions for loading a trained AutoEncoder and converting
between image space and latent space.

This module provides helper functions to:

- Load a trained AutoEncoder model
- Encode an image into its latent representation
- Decode a latent vector into a reconstructed image

These utilities simplify the interaction between PyTorch tensors and
NumPy arrays, which we need for the genetic algorithm. 

Notes
-----
All computations automatically run on GPU if available, otherwise on CPU.

Expected tensor shapes:

- Image tensor: (batch_size, 1, 64, 64)
- Latent vector: (batch_size, latent_dim)

Examples
--------
Load a trained model:

>>> model, latent_dim = load_model("autoencoder_olivetti.pth")

Encode an image:

>>> z = encode_image(model, image_tensor)

Decode a latent vector:

>>> image = decode_latent(model, z)
"""

def load_model(path):
    """
    Load a trained AutoEncoder model from a checkpoint file.

    Parameters
    ----------
    path : str
        Path to the saved model checkpoint (.pth file).

    Returns
    -------
    model : AutoEncoder
        Loaded AutoEncoder model in evaluation mode.

    latent_dim : int
        Dimension of the latent space used during training.

    Notes
    -----
    The checkpoint must contain:

    - "model_state_dict" : trained weights
    - "latent_dim" : latent space dimension
    """
    checkpoint = torch.load(path, map_location=device)
    latent_dim = checkpoint["latent_dim"]

    model = AutoEncoder(latent_dim).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, latent_dim


def encode_image(model, image_tensor):
    """
    Encode an image tensor into its latent representation.

    Parameters
    ----------
    model : AutoEncoder
        Trained autoencoder model.

    image_tensor : torch.Tensor
        Input tensor representing the image.
        Expected shape: (batch_size, 1, 64, 64).

    Returns
    -------
    numpy.ndarray
        Latent representation of the image with shape
        (batch_size, latent_dim) = (number of samples, size latent vector for each sample)
        2-dimensional NumPy array where each row corresponds to one encoded sample
        and each column corresponds to one latent featur. 

    Notes
    -----
    Gradient computation is disabled during encoding.We don't want to modify the weights anymore.
    """
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        z = model.encode(image_tensor)
    return z.cpu().numpy()


def decode_latent(model, latent_numpy):
    """
    Decode a latent vector into an image reconstruction.

    Parameters
    ----------
    model : AutoEncoder
        Trained autoencoder model.

    latent_numpy : numpy.ndarray
        Latent vector or batch of latent vectors with shape
        (batch_size, latent_dim).

    Returns
    -------
    torch.Tensor
        Reconstructed image tensor with shape
        (batch_size, 1, 64, 64).

    Notes
    -----
    The latent vector is converted to a PyTorch tensor before being
    passed through the decoder network.
    """
    with torch.no_grad():
        z = torch.tensor(latent_numpy, dtype=torch.float32).to(device)
        x = model.decode(z)
    return x.cpu()

##############################################################################
# ALGORITHME GENETIQUE: FUSION + MUTATION
# OUTPUT: vecteur fusionné/muté
##############################################################################

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




