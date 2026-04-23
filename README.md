# 4BiM - Interactive Forensic Composite Generator

**Authors**: Nathanael Pons, Axel Mondesir, Ambre Lesurtel, Skander Zouhir, Eglantine Salomé

## Overview

4BiM is a highly interactive application that visually reconstructs human faces via a **Conditional Variational Autoencoder (CVAE)** intertwined with a **Genetic Algorithm**. The software renders 10 real-time latent samples for the user, allowing them to explicitly fuse biological geometry and apply severe semantic variations iteratively until a perfect facial reconstruction is achieved.

## System Architecture

The repository follows a clean, modular Model-View-Controller paradigm explicitly separating the mathematical AI algorithms from the user interface rendering paths.

```text
ProjetBIM/
├── programme/                   # The Main Application Module
│   ├── main.py                  # Entry Point (Bootstraps virtual environment & loads App)
│   └── src/
│       ├── core/                # AI Core Library
│       │   ├── logic.py         # Mathematically fuses latent spaces & maintains spatial vectors
│       │   ├── algorithme_genetique.py # Manages matrix permutations and vector reconstruction
│       │   └── CVAE_architecture.py    # Deep Learning network topology map
│       ├── ui/                  # User Interface Implementation
│       │   ├── app.py           # Main window, state logic, and multi-thread controller
│       │   ├── screens.py       # Views (Grid renderers, Media Library, Custom Image Alignment)
│       │   ├── widgets.py       # Custom Glassmorphism UI components
│       │   ├── utils.py         # Event dispatching and thread utilities
│       │   └── constants.py     # Theming variables and UI grid dimensions
│       └── assets/              
│           └── vae_celeba_final.pth # Pre-trained Machine Learning Weights (Imported at runtime)
│
├── data/                        # Reference Datasets
│   ├── celeba/                  # Raw image dataset providing randomized facial seeds
│   ├── attribute_directions.npz # Derived High-Dimensional math limits for semantics (Smiling, etc)
│   └── attribute_vectors.npz    
│
├── research/                    # Academic Sandbox (Isolated)
│   ├── CVAE_train.py            # Development scripts for GPU training on the original model
│   └── ...                      # (Not required for final execution)
│
└── docs/                        # Reference Manuals (Sphinx & Agile methodology)
```

## Advanced Mathematical Processing

### Norm-Preserving Hypersphere Fusion
In a standard Gaussian VAE, linearly interpolating two faces mathematically scales the resulting tensor down, causing the output image to degenerate into a blurry structure. This system integrates **Norm-Preserving Algebraic Injection**, effectively calculating the input vector radii and snapping the fused face specifically onto the surface boundary of the hypersphere. Fusions retain 100% of parent HD structural sharpness.

### Forced Alignment Extraction (`CropDialog`)
Any external custom image introduced to the VAE model will inherently corrupt the vector space if not identical in geometric spacing. 4BiM introduces an Interactive Mask GUI utilizing CelebA boundary laws (forcing eyes squarely onto a `51% Y` threshold limit) allowing any arbitrary imported face to encode natively and sharply.

### Dynamic Attribute Escalations
Instead of statically pushing geometry variables across all semantics, modifiers scale dynamically representing their true physical cost. (e.g. `Baldness` requires a `6.5x` latent push to completely empty the hair coordinate plane compared to `3.0x` for `Smiling`).

## Installation & Execution

The software self-isolates on execution, avoiding system pollution.

```bash
# Simply execute the bootloader
# (It will load .venv autonomously alongside PyTorch dependencies)
python3 programme/main.py
```

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
