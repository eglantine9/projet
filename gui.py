"""
Interface graphique – fusion d'images par auto-encodeur.
Style Apple Glass (macOS), sidebar verticale, design moderne.
"""

import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw, ImageFilter
import numpy as np
import os
import zipfile
from datetime import datetime


# =============================================================================
# =============================================================================

FOND_PRINCIPAL = "#f0f0f5"
FOND_VERRE = "#ffffffcc"
FOND_SIDEBAR = "#1c1c1e"
TEXTE_PRINCIPAL = "#1d1d1f"
TEXTE_SECONDAIRE = "#86868b"
TEXTE_SIDEBAR = "#ffffff"
TEXTE_SIDEBAR_DIM = "#98989d"
ACCENT = "#007AFF"
ACCENT_HOVER = "#0056CC"
SELECTION_BORDURE = "#FF3B30"
CARTE_FOND = "#ffffff"
CARTE_BORDURE = "#e5e5ea"
SEPARATEUR = "#d1d1d6"
BOUTON_VERRE = "#e8e8ed"
BOUTON_VERRE_HOVER = "#d1d1d6"
SIDEBAR_HOVER = "#2c2c2e"
SIDEBAR_ACTIF = "#3a3a3c"
OMBRE = "#c7c7cc"

LARGEUR_FENETRE = 1100
HAUTEUR_FENETRE = 720
LARGEUR_SIDEBAR = 64
TAILLE_VIGNETTE = 110
COLONNES_GRILLE = 5
POLICE_TITRE = ("SF Pro Display", 26, "bold")
POLICE_SOUS_TITRE = ("SF Pro Text", 13)
POLICE_BOUTON = ("SF Pro Text", 12, "bold")
POLICE_PETIT = ("SF Pro Text", 11)
POLICE_SIDEBAR = ("SF Pro Text", 9)
POLICE_ICONE = ("SF Pro Text", 22)
RAYON_COIN = 12


# =============================================================================
# Utilitaires graphiques
# =============================================================================

def creer_image_arrondie(image_pil, taille, rayon=RAYON_COIN):
    """Applique un masque arrondi sur une image PIL."""
    image_pil = image_pil.resize((taille, taille), Image.LANCZOS)
    masque = Image.new("L", (taille, taille), 0)
    dessin = ImageDraw.Draw(masque)
    dessin.rounded_rectangle([(0, 0), (taille - 1, taille - 1)], rayon, fill=255)
    sortie = Image.new("RGBA", (taille, taille), (0, 0, 0, 0))
    sortie.paste(image_pil.convert("RGBA"), mask=masque)
    return sortie


def numpy_vers_photo(image_np, taille=TAILLE_VIGNETTE):
    """Convertit un array numpy (0-1, 64×64) en PhotoImage avec coins arrondis."""
    image_uint8 = (np.clip(image_np, 0, 1) * 255).astype(np.uint8)
    image_pil = Image.fromarray(image_uint8, mode="L").convert("RGB")
    image_arrondie = creer_image_arrondie(image_pil, taille)
    return ImageTk.PhotoImage(image_arrondie)


# =============================================================================
# Application principale
# =============================================================================

class Application(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Fusion d'images")
        self.geometry(f"{LARGEUR_FENETRE}x{HAUTEUR_FENETRE}")
        self.configure(bg=FOND_PRINCIPAL)
        self.resizable(False, False)

        # Données
        self.images_echantillon = []
        self.indices_selectionnes = []
        self.images_fusionnees = []
        self.indices_resultats_selectionnes = []

        # Structure : sidebar + zone principale
        self.cadre_racine = tk.Frame(self, bg=FOND_PRINCIPAL)
        self.cadre_racine.pack(fill="both", expand=True)

        self._creer_sidebar()

        # Zone de contenu principal
        self.zone_contenu = tk.Frame(self.cadre_racine, bg=FOND_PRINCIPAL)
        self.zone_contenu.pack(side="left", fill="both", expand=True)

        # Écrans empilés
        self.ecrans = {}
        for Classe in (EcranAccueil, EcranSelection, EcranResultat):
            ecran = Classe(parent=self.zone_contenu, controleur=self)
            self.ecrans[Classe.__name__] = ecran
            ecran.place(relwidth=1, relheight=1)

        self.afficher_ecran("EcranAccueil")

    def _creer_sidebar(self):
        """Barre latérale avec icônes vectorielles."""
        barre = tk.Frame(
            self.cadre_racine, bg=FOND_SIDEBAR,
            width=LARGEUR_SIDEBAR, height=HAUTEUR_FENETRE
        )
        barre.pack(side="left", fill="y")
        barre.pack_propagate(False)

        tk.Frame(barre, height=20, bg=FOND_SIDEBAR).pack()

        # Logo app
        canvas_logo = tk.Canvas(barre, width=36, height=36, bg=FOND_SIDEBAR, highlightthickness=0)
        canvas_logo.pack(pady=(10, 30))
        canvas_logo.create_oval(4, 4, 32, 32, fill=ACCENT, outline="")
        canvas_logo.create_oval(12, 12, 24, 24, fill="#ffffff", outline="")
        canvas_logo.create_oval(15, 15, 21, 21, fill=ACCENT, outline="")

        # Bouton Wiki
        self._creer_bouton_sidebar(barre, self._dessiner_icone_wiki, "Wiki", self._ouvrir_wiki)

        tk.Frame(barre, bg=FOND_SIDEBAR).pack(fill="y", expand=True)

        tk.Frame(barre, height=1, bg="#3a3a3c").pack(fill="x", padx=10, pady=5)

        # Bouton Réglages
        self._creer_bouton_sidebar(barre, self._dessiner_icone_reglages, "Réglages", self._ouvrir_parametres)

        tk.Frame(barre, height=15, bg=FOND_SIDEBAR).pack()

    @staticmethod
    def _dessiner_icone_wiki(canvas):
        """Icône livre dessinée en vectoriel."""
        c = canvas
        col = TEXTE_SIDEBAR
        # Page gauche
        c.create_polygon(10, 6, 20, 8, 20, 26, 10, 24, fill="", outline=col, width=1.5)
        # Page droite
        c.create_polygon(20, 8, 30, 6, 30, 24, 20, 26, fill="", outline=col, width=1.5)
        # Lignes de texte gauche
        for y in (12, 16, 20):
            c.create_line(13, y, 18, y + 0.5, fill=col, width=1)
        # Lignes de texte droite
        for y in (12, 16, 20):
            c.create_line(22, y + 0.5, 27, y, fill=col, width=1)

    @staticmethod
    def _dessiner_icone_reglages(canvas):
        """Icône engrenage dessinée en vectoriel."""
        import math
        c = canvas
        col = TEXTE_SIDEBAR
        cx, cy, r_ext, r_int = 20, 16, 11, 7
        dents = 8
        points = []
        for i in range(dents * 2):
            angle = math.pi * i / dents - math.pi / 2
            r = r_ext if i % 2 == 0 else r_int
            points.extend([cx + r * math.cos(angle), cy + r * math.sin(angle)])
        c.create_polygon(points, fill="", outline=col, width=1.5)
        c.create_oval(cx - 4, cy - 4, cx + 4, cy + 4, fill="", outline=col, width=1.5)

    def _creer_bouton_sidebar(self, parent, dessiner_icone, texte, commande):
        """Crée un bouton avec icône Canvas dans la sidebar."""
        cadre = tk.Frame(parent, bg=FOND_SIDEBAR, cursor="hand2")
        cadre.pack(pady=4, padx=6, fill="x")

        canvas_icone = tk.Canvas(cadre, width=40, height=32, bg=FOND_SIDEBAR, highlightthickness=0)
        canvas_icone.pack(pady=(6, 0))
        dessiner_icone(canvas_icone)

        label_texte = tk.Label(
            cadre, text=texte, font=POLICE_SIDEBAR,
            fg=TEXTE_SIDEBAR_DIM, bg=FOND_SIDEBAR
        )
        label_texte.pack(pady=(0, 6))

        def entrer(e):
            cadre.config(bg=SIDEBAR_HOVER)
            canvas_icone.config(bg=SIDEBAR_HOVER)
            label_texte.config(bg=SIDEBAR_HOVER)

        def sortir(e):
            cadre.config(bg=FOND_SIDEBAR)
            canvas_icone.config(bg=FOND_SIDEBAR)
            label_texte.config(bg=FOND_SIDEBAR)

        for widget in (cadre, canvas_icone, label_texte):
            widget.bind("<Enter>", entrer)
            widget.bind("<Leave>", sortir)
            widget.bind("<Button-1>", lambda e, c=commande: c())

    def afficher_ecran(self, nom):
        ecran = self.ecrans[nom]
        ecran.tkraise()
        if hasattr(ecran, "lors_affichage"):
            ecran.lors_affichage()

    def charger_echantillon(self):
        """Charge 10 visages depuis le dataset Olivetti."""
        from sklearn.datasets import fetch_olivetti_faces
        donnees = fetch_olivetti_faces(shuffle=True, random_state=42)
        indices = np.random.RandomState(0).choice(len(donnees.images), size=10, replace=False)
        self.images_echantillon = [donnees.images[i] for i in indices]

    def lancer_fusion(self):
        """
        Point d'entrée de la fusion.
        Branchement encodeur + algo génétique + décodeur à faire ici.
        """
        nombre_resultats = max(len(self.indices_selectionnes), 4)
        self.images_fusionnees = [
            np.random.rand(64, 64).astype(np.float32)
            for _ in range(nombre_resultats)
        ]

    def sauvegarder_resultats(self):
        """Sauvegarde les images sélectionnées en PNG + archive ZIP."""
        if not self.indices_resultats_selectionnes:
            messagebox.showwarning("Aucune sélection", "Sélectionnez au moins une image.")
            return

        dossier = filedialog.askdirectory(title="Dossier de sauvegarde")
        if not dossier:
            return

        horodatage = datetime.now().strftime("%Y%m%d_%H%M%S")
        dossier_sortie = os.path.join(dossier, f"fusion_{horodatage}")
        os.makedirs(dossier_sortie, exist_ok=True)

        chemins = []
        for idx in self.indices_resultats_selectionnes:
            img = (self.images_fusionnees[idx] * 255).astype(np.uint8)
            pil = Image.fromarray(img, mode="L")
            chemin = os.path.join(dossier_sortie, f"resultat_{idx + 1}.png")
            pil.save(chemin)
            chemins.append(chemin)

        chemin_zip = os.path.join(dossier, f"fusion_{horodatage}.zip")
        with zipfile.ZipFile(chemin_zip, "w") as zf:
            for c in chemins:
                zf.write(c, os.path.basename(c))

        messagebox.showinfo(
            "Sauvegarde terminée",
            f"{len(chemins)} image(s) → {dossier_sortie}\nArchive → {chemin_zip}"
        )

    def _ouvrir_wiki(self):
        messagebox.showinfo("Wiki", "Fonctionnalité à venir.")

    def _ouvrir_parametres(self):
        messagebox.showinfo("Réglages", "Fonctionnalité à venir.")


# =============================================================================
# Bouton Apple Glass
# =============================================================================

class BoutonVerre(tk.Canvas):
    """Bouton avec effet verre dépoli arrondi."""

    def __init__(self, parent, texte, commande, largeur=200, hauteur=42,
                 couleur_fond=BOUTON_VERRE, couleur_texte=TEXTE_PRINCIPAL,
                 est_accent=False):
        super().__init__(
            parent, width=largeur, height=hauteur,
            bg=parent["bg"], highlightthickness=0, cursor="hand2"
        )
        self.commande = commande
        self.texte = texte
        self.largeur = largeur
        self.hauteur = hauteur
        self.couleur_base = ACCENT if est_accent else couleur_fond
        self.couleur_hover = ACCENT_HOVER if est_accent else BOUTON_VERRE_HOVER
        self.couleur_texte = "#ffffff" if est_accent else couleur_texte

        self._dessiner(self.couleur_base)

        self.bind("<Enter>", lambda e: self._dessiner(self.couleur_hover))
        self.bind("<Leave>", lambda e: self._dessiner(self.couleur_base))
        self.bind("<Button-1>", lambda e: self.commande())

    def _dessiner(self, couleur):
        self.delete("all")
        r = 10
        l, h = self.largeur, self.hauteur
        # Rectangle arrondi
        self.create_arc(0, 0, r * 2, r * 2, start=90, extent=90, fill=couleur, outline="")
        self.create_arc(l - r * 2, 0, l, r * 2, start=0, extent=90, fill=couleur, outline="")
        self.create_arc(0, h - r * 2, r * 2, h, start=180, extent=90, fill=couleur, outline="")
        self.create_arc(l - r * 2, h - r * 2, l, h, start=270, extent=90, fill=couleur, outline="")
        self.create_rectangle(r, 0, l - r, h, fill=couleur, outline="")
        self.create_rectangle(0, r, l, h - r, fill=couleur, outline="")
        # Texte centré
        self.create_text(
            l / 2, h / 2, text=self.texte,
            fill=self.couleur_texte, font=POLICE_BOUTON
        )


# =============================================================================
# Carte image cliquable
# =============================================================================

class CarteImage(tk.Frame):
    """Vignette avec coins arrondis, ombre et bordure de sélection."""

    def __init__(self, parent, image_np, indice, callback, taille=TAILLE_VIGNETTE):
        super().__init__(parent, bg=parent["bg"])
        self.indice = indice
        self.est_selectionnee = False
        self.callback = callback
        self.taille = taille

        # Cadre avec ombre simulée
        self.cadre_ombre = tk.Frame(self, bg=OMBRE, padx=1, pady=1)
        self.cadre_ombre.pack()

        self.cadre_carte = tk.Frame(
            self.cadre_ombre, bg=CARTE_FOND, padx=3, pady=3
        )
        self.cadre_carte.pack()

        # Image
        self.photo = numpy_vers_photo(image_np, taille)
        self.label_image = tk.Label(
            self.cadre_carte, image=self.photo,
            bg=CARTE_FOND, cursor="hand2"
        )
        self.label_image.pack()

        # Clic
        self.label_image.bind("<Button-1>", self._clic)

    def _clic(self, event):
        self.est_selectionnee = not self.est_selectionnee
        self._maj_style()
        self.callback(self.indice, self.est_selectionnee)

    def _maj_style(self):
        if self.est_selectionnee:
            self.cadre_ombre.config(bg=SELECTION_BORDURE)
            self.cadre_carte.config(bg=SELECTION_BORDURE)
        else:
            self.cadre_ombre.config(bg=OMBRE)
            self.cadre_carte.config(bg=CARTE_FOND)


# =============================================================================
# US#A / US#B — Accueil
# =============================================================================

class EcranAccueil(tk.Frame):
    def __init__(self, parent, controleur):
        super().__init__(parent, bg=FOND_PRINCIPAL)
        self.controleur = controleur

        # Centrage vertical
        conteneur = tk.Frame(self, bg=FOND_PRINCIPAL)
        conteneur.place(relx=0.5, rely=0.45, anchor="center")

        # Icône décorative
        canvas_icone = tk.Canvas(
            conteneur, width=80, height=80,
            bg=FOND_PRINCIPAL, highlightthickness=0
        )
        canvas_icone.pack(pady=(0, 24))
        # Dégradé simulé avec cercles concentriques
        for i in range(40, 0, -1):
            ratio = i / 40
            r = int(0 + (0 * (1 - ratio)))
            g = int(122 * ratio + 86 * (1 - ratio))
            b = int(255 * ratio + 179 * (1 - ratio))
            c = f"#{r:02x}{g:02x}{b:02x}"
            canvas_icone.create_oval(
                40 - i, 40 - i, 40 + i, 40 + i,
                fill=c, outline=""
            )
        # Deux silhouettes qui se chevauchent (fusion)
        canvas_icone.create_oval(24, 20, 40, 36, fill="", outline="white", width=2)
        canvas_icone.create_oval(40, 20, 56, 36, fill="", outline="white", width=2)
        canvas_icone.create_oval(30, 38, 50, 55, fill="", outline="white", width=2)

        # Titre
        tk.Label(
            conteneur, text="Fusion d'images",
            font=("SF Pro Display", 32, "bold"),
            fg=TEXTE_PRINCIPAL, bg=FOND_PRINCIPAL
        ).pack(pady=(0, 8))

        # Description
        tk.Label(
            conteneur,
            text="Sélectionnez des visages, fusionnez-les par\nauto-encodeur et algorithme génétique.",
            font=POLICE_SOUS_TITRE, fg=TEXTE_SECONDAIRE,
            bg=FOND_PRINCIPAL, justify="center"
        ).pack(pady=(0, 36))

        # Bouton principal
        BoutonVerre(
            conteneur,
            texte="Commencer",
            commande=self._demarrer,
            largeur=220, hauteur=46,
            est_accent=True
        ).pack()

    def _demarrer(self):
        self.controleur.charger_echantillon()
        self.controleur.afficher_ecran("EcranSelection")


# =============================================================================
# US#C / US#D / US#E / US#F — Sélection
# =============================================================================

class EcranSelection(tk.Frame):
    def __init__(self, parent, controleur):
        super().__init__(parent, bg=FOND_PRINCIPAL)
        self.controleur = controleur
        self.cartes = []

        # En-tête
        entete = tk.Frame(self, bg=FOND_PRINCIPAL)
        entete.pack(fill="x", padx=36, pady=(28, 0))

        tk.Label(
            entete, text="Sélection",
            font=POLICE_TITRE, fg=TEXTE_PRINCIPAL, bg=FOND_PRINCIPAL
        ).pack(side="left")

        self.label_compteur = tk.Label(
            entete, text="0 sélectionnée(s)",
            font=POLICE_PETIT, fg=TEXTE_SECONDAIRE, bg=FOND_PRINCIPAL
        )
        self.label_compteur.pack(side="right")

        # Sous-titre
        tk.Label(
            self, text="Choisissez les visages à fusionner",
            font=POLICE_SOUS_TITRE, fg=TEXTE_SECONDAIRE,
            bg=FOND_PRINCIPAL, anchor="w"
        ).pack(fill="x", padx=36, pady=(4, 16))

        # Séparateur
        tk.Frame(self, height=1, bg=SEPARATEUR).pack(fill="x", padx=36)

        # Grille
        self.cadre_grille = tk.Frame(self, bg=FOND_PRINCIPAL)
        self.cadre_grille.pack(expand=True, pady=16)

        # Barre d'actions
        barre = tk.Frame(self, bg=FOND_PRINCIPAL)
        barre.pack(fill="x", padx=36, pady=(0, 24))

        BoutonVerre(
            barre, texte="← Retour", commande=self._retour,
            largeur=120, hauteur=38
        ).pack(side="left")

        BoutonVerre(
            barre, texte="Fusionner →", commande=self._fusionner,
            largeur=160, hauteur=38, est_accent=True
        ).pack(side="right")

    def lors_affichage(self):
        self.controleur.indices_selectionnes.clear()
        self._construire_grille()
        self._maj_compteur()

    def _construire_grille(self):
        for w in self.cadre_grille.winfo_children():
            w.destroy()
        self.cartes.clear()

        for i, img in enumerate(self.controleur.images_echantillon):
            carte = CarteImage(self.cadre_grille, img, i, self._clic_carte)
            carte.grid(
                row=i // COLONNES_GRILLE, column=i % COLONNES_GRILLE,
                padx=10, pady=10
            )
            self.cartes.append(carte)

    def _clic_carte(self, indice, selectionnee):
        if selectionnee and indice not in self.controleur.indices_selectionnes:
            self.controleur.indices_selectionnes.append(indice)
        elif not selectionnee and indice in self.controleur.indices_selectionnes:
            self.controleur.indices_selectionnes.remove(indice)
        self._maj_compteur()

    def _maj_compteur(self):
        n = len(self.controleur.indices_selectionnes)
        self.label_compteur.config(text=f"{n} sélectionnée(s)")

    def _fusionner(self):
        if len(self.controleur.indices_selectionnes) < 2:
            messagebox.showwarning("Sélection", "Sélectionnez au moins 2 images.")
            return
        self.controleur.lancer_fusion()
        self.controleur.afficher_ecran("EcranResultat")

    def _retour(self):
        self.controleur.afficher_ecran("EcranAccueil")


# =============================================================================
# US#G / US#H — Résultats
# =============================================================================

class EcranResultat(tk.Frame):
    def __init__(self, parent, controleur):
        super().__init__(parent, bg=FOND_PRINCIPAL)
        self.controleur = controleur
        self.cartes_res = []

        # En-tête
        entete = tk.Frame(self, bg=FOND_PRINCIPAL)
        entete.pack(fill="x", padx=36, pady=(28, 0))

        tk.Label(
            entete, text="Résultats",
            font=POLICE_TITRE, fg=TEXTE_PRINCIPAL, bg=FOND_PRINCIPAL
        ).pack(side="left")

        self.label_compteur = tk.Label(
            entete, text="0 sélectionnée(s)",
            font=POLICE_PETIT, fg=TEXTE_SECONDAIRE, bg=FOND_PRINCIPAL
        )
        self.label_compteur.pack(side="right")

        tk.Label(
            self, text="Sélectionnez les images à sauvegarder",
            font=POLICE_SOUS_TITRE, fg=TEXTE_SECONDAIRE,
            bg=FOND_PRINCIPAL, anchor="w"
        ).pack(fill="x", padx=36, pady=(4, 16))

        tk.Frame(self, height=1, bg=SEPARATEUR).pack(fill="x", padx=36)

        # Grille
        self.cadre_grille = tk.Frame(self, bg=FOND_PRINCIPAL)
        self.cadre_grille.pack(expand=True, pady=16)

        # Barre d'actions
        barre = tk.Frame(self, bg=FOND_PRINCIPAL)
        barre.pack(fill="x", padx=36, pady=(0, 24))

        BoutonVerre(
            barre, texte="↻ Recommencer", commande=self._recommencer,
            largeur=160, hauteur=38
        ).pack(side="left")

        BoutonVerre(
            barre, texte="Sauvegarder et terminer", commande=self._sauvegarder,
            largeur=220, hauteur=38, est_accent=True
        ).pack(side="right")

    def lors_affichage(self):
        self.controleur.indices_resultats_selectionnes.clear()
        self._construire_grille()
        self._maj_compteur()

    def _construire_grille(self):
        for w in self.cadre_grille.winfo_children():
            w.destroy()
        self.cartes_res.clear()

        for i, img in enumerate(self.controleur.images_fusionnees):
            carte = CarteImage(self.cadre_grille, img, i, self._clic_resultat)
            carte.grid(
                row=i // COLONNES_GRILLE, column=i % COLONNES_GRILLE,
                padx=10, pady=10
            )
            self.cartes_res.append(carte)

    def _clic_resultat(self, indice, selectionnee):
        if selectionnee and indice not in self.controleur.indices_resultats_selectionnes:
            self.controleur.indices_resultats_selectionnes.append(indice)
        elif not selectionnee and indice in self.controleur.indices_resultats_selectionnes:
            self.controleur.indices_resultats_selectionnes.remove(indice)
        self._maj_compteur()

    def _maj_compteur(self):
        n = len(self.controleur.indices_resultats_selectionnes)
        self.label_compteur.config(text=f"{n} sélectionnée(s)")

    def _sauvegarder(self):
        self.controleur.sauvegarder_resultats()

    def _recommencer(self):
        self.controleur.charger_echantillon()
        self.controleur.afficher_ecran("EcranSelection")


# =============================================================================
# Point d'entrée
# =============================================================================

if __name__ == "__main__":
    app = Application()
    app.mainloop()
