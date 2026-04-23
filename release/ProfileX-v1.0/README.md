# ProfileX - Interactive Forensic Composite Generator

**Authors**: Nathanael Pons, Axel Mondesir, Ambre Lesurtel, Skander Zouhir, Eglantine Salomé

## Overview

ProfileX is a highly interactive application that visually reconstructs human faces via a **Conditional Variational Autoencoder (CVAE)** intertwined with a **Genetic Algorithm**. The software renders 10 real-time latent samples for the user, allowing them to explicitly fuse biological geometry and apply severe semantic variations iteratively until a perfect facial reconstruction is achieved.

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
Any external custom image introduced to the VAE model will inherently corrupt the vector space if not identical in geometric spacing. ProfileX introduces an Interactive Mask GUI utilizing CelebA boundary laws (forcing eyes squarely onto a `51% Y` threshold limit) allowing any arbitrary imported face to encode natively and sharply.

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

## Usage

To start the application, run the main script from the root:

```bash
python programme/main.py
```

Upon launch, the user is initially presented with 10 images. They can select the ones that look closest to their target (the person being searched for) in order to fuse and mutate them. A second group of images resulting from these operations will then be proposed, and so on until satisfaction.

The user can also save the resulting images, import their own images, or restart the entire process.

Several parameters can be controlled from the interface. Here is the full list of features:

### Features

* **Select** : By clicking on an image, the user selects it. Selected images are highlighted with a red border.
* **Merge** : By clicking on the "Fuse →" button, the user confirms the selected images. The software merges the latent space of all selected images, displays the result of this fusion in the top left corner, and displays 9 other mutant images generated from this fusion.
* **Save** : By clicking on "Save", the user stores the chosen image(s) on their device, either in PNG format or grouped in a ZIP archive.
* **Back** : Cancels operations and returns to the previous set of images.

### Algorithm Settings

In the settings section, it is possible to modify the impact of the genetic algorithm:
* **Mutation rate** : The amount of traits or regions mutated between each generation.
* **Mutation level / intensity** : The amplitude or force of the mutations applied to the latent space (measured visual impact).
* **Number of images / Variant Count** : The number of mutant images proposed from the initial fusion.
