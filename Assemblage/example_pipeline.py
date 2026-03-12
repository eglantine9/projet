######################################################
# TEST PIPELINE OLIVETTI AUTOENCODER
# exemple de script pour faire appel à functions_library 
# Output: encodage + mutation + fusion + reconstruction + visualisation
# fonctionne en lançant dans le terminal "python3 example_pipeline.py 
# UNIQUEMENT si dans le même dossier que functions_library.py et autoencodeur entraîné (fichier pth)
######################################################


import torch
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from functions_library import load_model, encode_image, decode_latent, fusion, mutation


# ======================
# Charger modèle
# ======================

model, latent_dim = load_model("autoencoder_olivetti.pth")
print("Latent dim =", latent_dim)


# ======================
# Charger images
# ======================

data = fetch_olivetti_faces(shuffle=True, random_state=42) # 42 pour fixer le générateur aléatoire
images = torch.tensor(data.images, dtype=torch.float32).unsqueeze(1)
sample_imgs = images[16:24] # il y a 400 images dans la base de données, j'ai décidé de prendre 8 images aléatoirs, celles de 8 à 16
# il faudra remplacer cet échantillon d'images par les images sélectionnées 


# ======================
# ENCODE
# ======================

latent_vectors = encode_image(model, sample_imgs)
print("Shape latent:", latent_vectors.shape)
print(type(latent_vectors))
print(latent_vectors)


# ======================
# FUSION
# ======================

# on fusionne les vecteurs latents obtenus précédemment
vecteur_fusion = fusion(latent_vectors)
print("Vecteur fusionné:", vecteur_fusion)


# ======================
# MUTATION
# ======================

vecteurs_mutants = mutation(
    vecteur_fusion,
    nb_variants=8,
    taux_mutation=0.2,
    niveau_mutation=0.5
)
print("Mutants shape:", vecteurs_mutants.shape)
print("Vecteurs mutés:", vecteurs_mutants)


# ======================
# DECODE
# ======================

# pour pouvoir afficher l'image fusionnée qui est unique 
vecteur_fusion_tensor = torch.tensor(vecteur_fusion, dtype=torch.float32).unsqueeze(0) # ajouter dimension batch car un seul vecteur renvoyé
image_fusion = decode_latent(model, vecteur_fusion_tensor.numpy())
image_fusion = image_fusion.squeeze() # image unique donc pas besoin de savoir nombre de batch

# pour pouvoir afficher l'image finale = fusionnée + mutée
# pas besoin de unsqueeze car au moins 2 vecteurs renvoyés
images_resultat = decode_latent(model, vecteurs_mutants) 
images_resultat = images_resultat.squeeze(1)


# ======================
# VISUALISATION
# ======================

images_init = sample_imgs.squeeze(1) # enlever dimension channel pour faciliter affichage matplotlib
images_mut = images_resultat
image_fus = image_fusion

n_init = images_init.shape[0]
n_mut = images_mut.shape[0]

n = max(n_init, n_mut, 1)

fig, axes = plt.subplots(3, n, figsize=(3*n, 9)) #3 lignes, n colonnes, (width (chaque image environ 3 pouces de largeur, donc 3*n), height (3*3lignes))

# cas particulier si n = 1
if n == 1:
    axes = axes.reshape(3,1) #3 lignes 1 colonne

# ----- Images initiales -----
for i in range(n_init):
    axes[0, i].imshow(images_init[i], cmap="gray") #axes[ligne, colonne]
    axes[0, i].axis("off") #cacher axes du graphique

for i in range(n_init, n):
    axes[0, i].axis("off")

# ----- Image fusionnée -----
axes[1, 0].imshow(image_fus, cmap="gray")
axes[1, 0].axis("off")

for i in range(1, n):
    axes[1, i].axis("off")

# ----- Images mutées -----
for i in range(n_mut):
    axes[2, i].imshow(images_mut[i], cmap="gray")
    axes[2, i].axis("off")

for i in range(n_mut, n):
    axes[2, i].axis("off")

# ----- Titres par ligne centrés -----
titles = ["Images initiales", "Fusion", "Mutations"]
for idx, title in enumerate(titles):
    fig.text(0.5, 0.93 - idx*0.3, title, ha='center', va='center', fontsize=14, fontweight='bold')
    #x=0.5 milieu de la figure à l'horizontale
    #y=0.95-idx-0.32 descendre petit = petit à la verticale
    # titres
    # alignement central = ha
    # alignement vertical = va
    


plt.suptitle("Pipeline génétique dans l'espace latent", fontsize=16)
plt.tight_layout()
plt.show()