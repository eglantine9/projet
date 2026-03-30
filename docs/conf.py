# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys

project = 'Projet 4BIM portaits Groupe 5'
copyright = '2026, Nathanael PONS, Axel MONDESIR, Ambre LESURTEL, Skander ZOUHIR, Eglantine SALOME'
author = 'Nathanael PONS, Axel MONDESIR, Ambre LESURTEL, Skander ZOUHIR, Eglantine SALOME'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc','sphinx.ext.doctest','sphinx.ext.todo','sphinx.ext.coverage','sphinx.ext.mathjax','sphinx.ext.ifconfig','sphinx.ext.viewcode','sphinx.ext.githubpages','sphinx.ext.napoleon']

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']

sys.path.insert(0,os.path.abspath("./"))
