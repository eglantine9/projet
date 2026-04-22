# 4BiM - Générateur de Portrait Robot Interactif

**Auteurs** : Nathanael Pons, Axel Mondesir, Ambre Lesurtel, Skander Zouhir, Eglantine Salomé

## Introduction

L'objectif de ce logiciel est de générer le portrait d'une personne (de type portrait-robot). Pour ce faire, une interface présente 10 images à l'utilisateur sur lesquelles il peut interagir. L'utilisateur peut y effectuer les actions qu'il souhaite jusqu'à l'obtention d'un résultat qui le satisfait.

Le projet repose sur la combinaison d'un **CVAE (Conditional Variational Autoencoder)** pour l'encodage et la génération d'images, couplé à un **Algorithme Génétique** qui gère la mutation et la fusion (croisement) des caractéristiques dans l'espace latent.

## Architecture Complète du Projet

Voici l'arborescence détaillée du projet, couvrant l'intégralité de la base de code, des ressources IA, et du logiciel :

### 1. `programme/` - Application et Interface Utilisateur
Code finalisé du logiciel interactif, intégrant les algorithmes avec une interface graphique moderne.
- **`main.py`** : Point d'entrée à exécuter pour lancer le logiciel.
- **`venv/`** : Environnement virtuel Python contenant localement les dépendances requises.
- **`Algorithme_ Genetique/`** : Scripts et fonctions historiques de l'algorithme génétique autonome utilisé dans le projet.
- **`src/`** : Code source modulaire de l'application.
  - **`assets/`** : Contient le poids final de l'IA chargé in-app (`vae_celeba_final.pth`).
  - **`core/`** : Logique métier et intelligence artificielle.
    - `models.py` : Architecture formelle du **CVAE** (`AutoEncoder`/`VAE`).
    - `logic.py` : Cœur algorithmique. Fait la jonction : encodage des visages, calculs matriciels des vecteurs latents (croisement/fusion), et applications des masques de mutation.
  - **`ui/`** : Composants de l'ensemble de l'Interface Utilisateur (développée en module Tkinter).
    - `app.py` : Fenêtre principale, state management et coordinateur des calculs. S'occupe du chargement aléatoire des échantillons depuis CelebA.
    - `constants.py` : Configuration de l'UI (thème couleur, dimensions).
    - `screens.py`, `widgets.py`, `utils.py` : Décomposition de l'interface en onglets et gestion des events.

### 2. `data/` - Base de données & Matrices Latentes
Dossier contenant l'épaisse base de visages et d'extrapolations statistiques.
- **`celeba/`** : Dataset brut.
  - `img_align_celeba/` : Contient l'intégralité des images (+200000) de visages de la base CelebA.
  - Fichiers `.txt` (`list_attr_celeba`, `identity_CelebA`...) : Méta-données et classifications annexes des visages (dimensions sémantiques, points clés).
- **`attribute_directions.npz`** & **`attribute_vectors.npz`** : Matrices de calcul pré-générées représentant des dimensions sémantiques (pour guider la mutation dans des axes comme des "lunettes" ou "genre" sur l'espace latent VAE).

### 3. `research/` - Environnement IA & Pipelines
Dossier isolé contenant le framework de Deep Learning PyTorch, destiné aux itérations et à la preuve de construction du VAE. Pas requis pour lancer le logiciel.
- **`CVAE_architecture.py`** : Brouillon original brut de l'architecture du CVAE (Encodeur / Décodeur / Forward avec Kullback-Leibler).
- **`CVAE_train.py`** : Script d'entraînement avec itérations Epoch, Backpropagation et affichage des Losses sur GPU.
- **`CVAE_test.py`** : Script de test post-entrainement mesurant la capacité du modèle construit.
- **`CVAE_latent.py`** : Utilitaires pour décortiquer un sous-espace d'encodage.
- **`CVAE_attributes_vectors.py`** & **`CVAE_attributs_directions.py`** : Pipeline utilisé pour produire les fichiers `.npz` de direction d'attributs sémantiques.

### 4. `docs/` - Documentation Officielle & Agilité
- Support technique au format **Sphinx** pour la production d’un site vitrine ou PDF `(index.rst`, `conf.py`, `Makefile`).
- **`ProductBacklog.pdf`** : Le cahier des charges et le plan Agile structurant la construction du projet BIM.

## Installation

### Prérequis

Assurez-vous d'avoir Python 3.x installé sur votre système. Vous aurez également besoin des bibliothèques suivantes :

- `torch` et `torchvision` (pour le modèle CVAE)
- `numpy` (pour les calculs matriciels)
- `Pillow` (pour la manipulation d'images dans l'interface)
- `scikit-learn` (utilisé pour charger les images initiales / d'entraînement)

Vous pouvez installer ces dépendances via la commande `pip` :

```bash
pip install torch torchvision numpy Pillow scikit-learn
```

## Utilisation

Pour démarrer l'application, exécutez le script principal depuis la racine :

```bash
python programme/main.py
```

Au lancement, l'utilisateur se voit initialement proposer 10 images. Il peut sélectionner celles qui lui semblent les plus proches de sa cible (la personne recherchée) afin de les fusionner et de les faire muter. Un deuxième groupe d'images résultant de ces opérations lui sera ensuite proposé, et ainsi de suite jusqu'à satisfaction.

L'utilisateur peut également sauvegarder les images obtenues, importer ses propres images, ou redémarrer le processus complet.

Plusieurs paramètres sont contrôlables depuis l'interface. Voici la liste complète des fonctionnalités :

### Fonctionnalités

* **Sélection (Select)** : En cliquant sur une image, l'utilisateur la sélectionne. Les images sélectionnées sont caractérisées par une bordure rouge.
* **Fusion (Merge)** : En cliquant sur le bouton « fusionner et muter », l'utilisateur confirme les images sélectionnées. Le logiciel fusionne l'espace latent de toutes les images sélectionnées, affiche le résultat de cette fusion dans le coin supérieur gauche, et affiche 9 autres images générées à partir de variations mutantes de cette fusion.
* **Sauvegarde (Save)** : En cliquant sur « Save », l'utilisateur stocke la ou les images choisies sur son appareil, soit au format PNG, soit regroupées dans une archive ZIP.
* **Retour (Back)** : Permet d'annuler les opérations et de revenir à l'ensemble d'images précédent.

### Paramètres de l'Algorithme (Settings)

Dans la section réglages, il est possible de modifier l'impact de l'algorithme génétique :
* **Taux de mutation (Mutation rate)** : La quantité de traits ou régions mutées entre chaque génération.
* **Intensité de mutation (Mutation intensity)** : L'amplitude ou la force des mutations appliquées sur l'espace latent (impact visuel mesuré).
* **Nombre d'images (Number of images / Variant Count)** : Le nombre d'images mutantes proposées à partir de la fusion initiale.
