"""Application screens: Home, Settings, Guide, Media, Credits, and Grid."""

import math
import os
import subprocess
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np

from .constants import *
from .widgets import GlassButton, ImageCard

# =============================================================================
# Home Screen
# =============================================================================

class HomeScreen(tk.Frame):
    """Landing screen with the application logo and a Get Started button."""

    def __init__(self, parent, controller):
        super().__init__(parent, bg=BG_MAIN)
        self.controller = controller

        container = tk.Frame(self, bg=BG_MAIN)
        container.place(relx=0.5, rely=0.45, anchor="center")

        icon_canvas = tk.Canvas(
            container, width=80, height=80, bg=BG_MAIN, highlightthickness=0
        )
        icon_canvas.pack(pady=(0, 24))
        for i in range(40, 0, -1):
            ratio = i / 40
            r = 0
            g = int(122 * ratio + 86 * (1 - ratio))
            b = int(255 * ratio + 179 * (1 - ratio))
            c = f"#{r:02x}{g:02x}{b:02x}"
            icon_canvas.create_oval(40 - i, 40 - i, 40 + i, 40 + i, fill=c, outline="")
        icon_canvas.create_oval(24, 20, 40, 36, fill="", outline="white", width=2)
        icon_canvas.create_oval(40, 20, 56, 36, fill="", outline="white", width=2)
        icon_canvas.create_oval(30, 38, 50, 55, fill="", outline="white", width=2)

        tk.Label(
            container, text="ProfileX",
            font=("SF Pro Display", 32, "bold"),
            fg=TEXT_PRIMARY, bg=BG_MAIN
        ).pack(pady=(0, 8))

        tk.Label(
            container,
            text="Select faces, merge them using an\nautoencoder and genetic algorithm.",
            font=FONT_SUBTITLE, fg=TEXT_SECONDARY,
            bg=BG_MAIN, justify="center"
        ).pack(pady=(0, 36))

        GlassButton(
            container, text="Get Started",
            command=self._start, width=220, height=46, is_accent=True
        ).pack()

    def _start(self):
        self.controller.load_sample()


# =============================================================================
# Settings Screen
# =============================================================================

class SettingsScreen(tk.Frame):
    """Settings panel for genetic algorithm parameters."""

    def __init__(self, parent, controller):
        super().__init__(parent, bg=BG_MAIN)
        self.controller = controller

        container = tk.Frame(self, bg=BG_MAIN)
        container.place(relx=0.5, rely=0.4, anchor="center")

        # Icon + Title
        title_frame = tk.Frame(container, bg=BG_MAIN)
        title_frame.pack(pady=(0, 4))
        ic = tk.Canvas(title_frame, width=32, height=32, bg=BG_MAIN, highlightthickness=0)
        ic.pack(side="left", padx=(0, 8))
        cx, cy, re, ri = 16, 16, 14, 9
        pts = []
        for ii in range(16):
            a = math.pi * ii / 8 - math.pi / 2
            r = re if ii % 2 == 0 else ri
            pts.extend([cx + r * math.cos(a), cy + r * math.sin(a)])
        ic.create_polygon(pts, fill="", outline=TEXT_SECONDARY, width=1.8)
        ic.create_oval(cx - 4, cy - 4, cx + 4, cy + 4, fill="", outline=TEXT_SECONDARY, width=1.8)
        tk.Label(
            title_frame, text="Settings",
            font=("SF Pro Display", 28, "bold"),
            fg=TEXT_PRIMARY, bg=BG_MAIN
        ).pack(side="left")

        tk.Label(
            container, text="Genetic algorithm parameters",
            font=FONT_SUBTITLE, fg=TEXT_SECONDARY, bg=BG_MAIN
        ).pack(pady=(0, 24))

        card = tk.Frame(container, bg=CARD_BG, padx=28, pady=20)
        card.pack()
        card.columnconfigure(1, weight=1)

        # Variants
        var_v = tk.IntVar(value=controller.variant_count)
        tk.Label(card, text="Variants", font=FONT_SUBTITLE,
                 fg=TEXT_PRIMARY, bg=CARD_BG).grid(row=0, column=0, sticky="w", pady=(6, 0))
        lbl_v = tk.Label(card, text=str(controller.variant_count),
                         font=("SF Pro Text", 12, "bold"), fg=ACCENT, bg=CARD_BG, width=4)
        lbl_v.grid(row=0, column=2, sticky="e", pady=(6, 0))
        tk.Scale(card, from_=1, to=20, orient="horizontal", variable=var_v,
                 showvalue=False, bg=CARD_BG, fg=ACCENT, troughcolor=BTN_GLASS,
                 highlightthickness=0, length=240,
                 command=lambda v: lbl_v.config(text=str(int(float(v))))
                 ).grid(row=0, column=1, sticky="ew", padx=8, pady=(6, 0))

        # Mutation rate
        var_r = tk.DoubleVar(value=controller.mutation_rate)
        tk.Label(card, text="Mutation Rate", font=FONT_SUBTITLE,
                 fg=TEXT_PRIMARY, bg=CARD_BG).grid(row=1, column=0, sticky="w", pady=(6, 0))
        lbl_r = tk.Label(card, text=f"{controller.mutation_rate:.2f}",
                         font=("SF Pro Text", 12, "bold"), fg=ACCENT, bg=CARD_BG, width=4)
        lbl_r.grid(row=1, column=2, sticky="e", pady=(6, 0))
        tk.Scale(card, from_=0.0, to=1.0, resolution=0.05, orient="horizontal", variable=var_r,
                 showvalue=False, bg=CARD_BG, fg=ACCENT, troughcolor=BTN_GLASS,
                 highlightthickness=0, length=240,
                 command=lambda v: lbl_r.config(text=f"{float(v):.2f}")
                 ).grid(row=1, column=1, sticky="ew", padx=8, pady=(6, 0))

        # Mutation level
        var_l = tk.IntVar(value=controller.mutation_level)
        tk.Label(card, text="Mutation Level", font=FONT_SUBTITLE,
                 fg=TEXT_PRIMARY, bg=CARD_BG).grid(row=2, column=0, sticky="w", pady=(6, 0))
        lbl_l = tk.Label(card, text=str(controller.mutation_level),
                         font=("SF Pro Text", 12, "bold"), fg=ACCENT, bg=CARD_BG, width=4)
        lbl_l.grid(row=2, column=2, sticky="e", pady=(6, 0))
        tk.Scale(card, from_=1, to=10, orient="horizontal", variable=var_l,
                 showvalue=False, bg=CARD_BG, fg=ACCENT, troughcolor=BTN_GLASS,
                 highlightthickness=0, length=240,
                 command=lambda v: lbl_l.config(text=str(int(float(v))))
                 ).grid(row=2, column=1, sticky="ew", padx=8, pady=(6, 0))

        # Buttons
        btn_frame = tk.Frame(container, bg=BG_MAIN)
        btn_frame.pack(fill="x", pady=(20, 0))

        def reset():
            var_v.set(DEFAULT_VARIANT_COUNT); lbl_v.config(text=str(DEFAULT_VARIANT_COUNT))
            var_r.set(DEFAULT_MUTATION_RATE); lbl_r.config(text=f"{DEFAULT_MUTATION_RATE:.2f}")
            var_l.set(DEFAULT_MUTATION_LEVEL); lbl_l.config(text=str(DEFAULT_MUTATION_LEVEL))

        def apply():
            controller.variant_count = var_v.get()
            controller.mutation_rate = var_r.get()
            controller.mutation_level = var_l.get()
            messagebox.showinfo("Settings", "Parameters applied.")

        GlassButton(btn_frame, text="Reset", command=reset, width=140, height=36).pack(side="left")
        GlassButton(btn_frame, text="Apply", command=apply,
                    width=140, height=36, is_accent=True).pack(side="right")


# =============================================================================
# Documentation Screen
# =============================================================================

class DocsScreen(tk.Frame):
    """Guide page embedded in a tab."""

    def __init__(self, parent, controller):
        super().__init__(parent, bg=BG_MAIN)
        self.controller = controller

        self.scroll_canvas = tk.Canvas(self, bg=BG_MAIN, highlightthickness=0)
        scrollbar = tk.Scrollbar(self, orient="vertical", command=self.scroll_canvas.yview)
        content = tk.Frame(self.scroll_canvas, bg=BG_MAIN)

        content.bind("<Configure>", lambda e: self.scroll_canvas.configure(scrollregion=self.scroll_canvas.bbox("all")))
        self.scroll_canvas.create_window((0, 0), window=content, anchor="nw")
        self.scroll_canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side="right", fill="y")
        self.scroll_canvas.pack(side="left", fill="both", expand=True)

        pad = 48

        # Icon + Title
        title_frame = tk.Frame(content, bg=BG_MAIN)
        title_frame.pack(fill="x", padx=pad, pady=(36, 4))
        ic = tk.Canvas(title_frame, width=32, height=32, bg=BG_MAIN, highlightthickness=0)
        ic.pack(side="left", padx=(0, 8))
        col_ic = TEXT_SECONDARY
        ic.create_polygon(6, 4, 16, 6, 16, 28, 6, 26, fill="", outline=col_ic, width=1.8)
        ic.create_polygon(16, 6, 26, 4, 26, 26, 16, 28, fill="", outline=col_ic, width=1.8)
        for yy in (10, 15, 20):
            ic.create_line(9, yy, 14, yy + 0.5, fill=col_ic, width=1)
            ic.create_line(18, yy + 0.5, 23, yy, fill=col_ic, width=1)
        tk.Label(
            title_frame, text="Guide",
            font=("SF Pro Display", 28, "bold"),
            fg=TEXT_PRIMARY, bg=BG_MAIN
        ).pack(side="left")

        tk.Label(
            content, text="How to use the ProfileX application",
            font=FONT_SUBTITLE, fg=TEXT_SECONDARY, bg=BG_MAIN, anchor="w"
        ).pack(fill="x", padx=pad, pady=(0, 20))

        tk.Frame(content, height=1, bg=SEPARATOR).pack(fill="x", padx=pad)

        sections = [
            ("How does it work?",
             "The application uses a Variational Autoencoder (VAE) trained on the\n"
             "CelebA face dataset. Each face is encoded into a 512-dimensional\n"
             "latent space, then latent vectors are combined by weighted fusion\n"
             "and mutation (genetic algorithm) to generate new faces."),
            ("Step 1 — Selection",
             "Click 'Get Started' to load a random sample of CelebA faces.\n"
             "Click on images to select them (red border = selected).\n"
             "Each selected image shows a Weight slider and Mirror toggle."),
            ("Step 2 — Weight & Mirror",
             "• Weight (0.0 – 2.0): controls how much influence an image has\n"
             "  in the fusion. Higher weight = more resemblance to that face.\n"
             "• Mirror: horizontally flips the image before encoding.\n"
             "  Useful when fusing faces looking in opposite directions."),
            ("Step 3 — Attributes",
             "Before fusing, toggle semantic attributes:\n"
             "• Bald: adds baldness to the result.\n"
             "• Eyeglasses: adds glasses to the result.\n"
             "• Smiling: makes the result smile.\n"
             "• Mustache: adds a mustache to the result.\n"
             "These use learned directions in the VAE latent space."),
            ("Step 4 — Fusion",
             "Click 'Fuse →' to start the process.\n"
             "• 1 selected image → generates variants by mutation only.\n"
             "• 2+ images → weighted fusion of latent vectors + mutation.\n"
             "The first result is the pure fusion; the rest are mutations."),
            ("Step 5 — Save",
             "Select the desired results and click 'Save'.\n"
             "Images are exported as PNG + ZIP archive in the chosen folder."),
            ("Settings",
             "• Variants: number of mutations generated (1 to 20).\n"
             "• Mutation Rate: probability that a gene is mutated (0 to 1).\n"
             "• Mutation Level: amplitude of mutation noise (1 to 10)."),
            ("My Media",
             "Import your own photos via the Media tab or the '+ Import' button.\n"
             "Imported images are automatically converted to 64×64 grayscale\n"
             "to match the VAE input format."),
        ]

        for title, body in sections:
            tk.Label(
                content, text=title, font=("SF Pro Display", 18, "bold"),
                fg=TEXT_PRIMARY, bg=BG_MAIN, anchor="w"
            ).pack(fill="x", padx=pad, pady=(24, 6))
            tk.Label(
                content, text=body, font=FONT_SUBTITLE, fg=TEXT_SECONDARY,
                bg=BG_MAIN, anchor="w", justify="left"
            ).pack(fill="x", padx=(pad + 8, pad))

        tk.Frame(content, height=40, bg=BG_MAIN).pack()


class CropDialog(tk.Toplevel):
    def __init__(self, parent, pil_image):
        super().__init__(parent)
        self.title("Extract Face (CelebA Alignment)")
        self.geometry("500x680")
        self.resizable(False, False)
        self.configure(bg=BG_MAIN)
        self.transient(parent)
        
        self.result_img = None
        self.src_img = pil_image
        self.src_w, self.src_h = self.src_img.size
        
        # Scale image to fit 400x400 max limits guaranteeing buttons never cut off
        self.scale = min(400 / max(1, self.src_w), 400 / max(1, self.src_h))
        self.disp_w = max(1, int(self.src_w * self.scale))
        self.disp_h = max(1, int(self.src_h * self.scale))
        
        self.disp_img = self.src_img.resize((self.disp_w, self.disp_h), Image.LANCZOS)
        self.tk_img = ImageTk.PhotoImage(self.disp_img)
        
        tk.Label(self, text="Align the mask exactly with the face!\nEyes on the green circles, mouth on the red line.",
                 font=FONT_SUBTITLE, fg=TEXT_PRIMARY, bg=BG_MAIN).pack(pady=10)
        
        self.canvas_frame = tk.Frame(self, bg=CARD_BG, highlightthickness=1, highlightbackground=SEPARATOR)
        self.canvas_frame.pack(pady=10)
        
        self.canvas = tk.Canvas(self.canvas_frame, width=self.disp_w, height=self.disp_h, bg=CARD_BG, highlightthickness=0, cursor="crosshair")
        self.canvas.pack()
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img)
        
        self.crop_size = min(self.disp_w, self.disp_h) // 2
        self.cx = self.disp_w // 2
        self.cy = self.disp_h // 2
        
        self.elements = []
        self._draw_mask()
        
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<Button-1>", self._on_drag)
        
        tk.Label(self, text="Adjust Size:", font=FONT_SMALL, fg=TEXT_SECONDARY, bg=BG_MAIN).pack()
        self.slider = tk.Scale(self, from_=20, to=min(self.disp_w, self.disp_h), orient="horizontal", 
                               command=self._on_scale, bg=BG_MAIN, fg=ACCENT, troughcolor=CARD_BG, highlightthickness=0)
        self.slider.set(self.crop_size)
        self.slider.pack(fill="x", padx=40, pady=5)
        
        GlassButton(self, "Crop & Import", self._confirm, width=150, height=40, is_accent=True).pack(pady=10)
        
        self.grab_set()
        self.wait_window(self)

    def _draw_mask(self):
        for e in self.elements:
            self.canvas.delete(e)
        self.elements.clear()
        
        r = self.crop_size // 2
        x0, y0 = self.cx - r, self.cy - r
        x1, y1 = self.cx + r, self.cy + r
        
        # Bounding box
        self.elements.append(self.canvas.create_rectangle(x0, y0, x1, y1, outline=ACCENT, width=2, dash=(4, 4)))
        
        # CelebA Anatomical Rules mapping
        C = self.crop_size
        
        # Eyes at exactly 51% y
        eye_y = y0 + C * 0.51
        eye_lx = x0 + C * 0.39
        eye_rx = x0 + C * 0.61
        er = max(2, C * 0.04) # eye radius
        
        self.elements.append(self.canvas.create_oval(eye_lx-er, eye_y-er, eye_lx+er, eye_y+er, outline="#32d74b", width=2))
        self.elements.append(self.canvas.create_oval(eye_rx-er, eye_y-er, eye_rx+er, eye_y+er, outline="#32d74b", width=2))
        
        # Nose at 66% y
        nose_y = y0 + C * 0.66
        nose_x = x0 + C * 0.5
        nr = max(1, C * 0.02)
        self.elements.append(self.canvas.create_oval(nose_x-nr, nose_y-nr, nose_x+nr, nose_y+nr, outline="#ff9f0a", width=2))
        
        # Mouth horizontal line at strictly 75% y
        mouth_y = y0 + C * 0.75
        mouth_w = C * 0.15
        self.elements.append(self.canvas.create_line(nose_x - mouth_w, mouth_y, nose_x + mouth_w, mouth_y, fill="#ff453a", width=3))
        
        # Head silhouette (approximate jawline constraint boundary)
        hx0, hy0 = x0 + C*0.15, y0 + C*0.1
        hx1, hy1 = x1 - C*0.15, y1 - C*0.05
        self.elements.append(self.canvas.create_oval(hx0, hy0, hx1, hy1, outline=ACCENT, width=1, dash=(2, 4)))

    def _on_drag(self, e):
        self.cx = max(0, min(self.disp_w, e.x))
        self.cy = max(0, min(self.disp_h, e.y))
        self._draw_mask()
        
    def _on_scale(self, v):
        self.crop_size = int(float(v))
        self._draw_mask()
        
    def _confirm(self):
        r = self.crop_size // 2
        left = int((self.cx - r) / self.scale)
        top = int((self.cy - r) / self.scale)
        right = int((self.cx + r) / self.scale)
        bottom = int((self.cy + r) / self.scale)
        
        self.result_img = self.src_img.crop((left, top, right, bottom))
        self.destroy()

# =============================================================================
# Media Library Screen
# =============================================================================

class MediaScreen(tk.Frame):
    """Custom photo library for importing and managing user images."""

    def __init__(self, parent, controller):
        super().__init__(parent, bg=BG_MAIN)
        self.controller = controller

        container = tk.Frame(self, bg=BG_MAIN)
        container.pack(fill="both", expand=True, padx=36, pady=28)

        header = tk.Frame(container, bg=BG_MAIN)
        header.pack(fill="x")

        # Icon + Title
        title_frame = tk.Frame(header, bg=BG_MAIN)
        title_frame.pack(side="left")
        ic = tk.Canvas(title_frame, width=32, height=32, bg=BG_MAIN, highlightthickness=0)
        ic.pack(side="left", padx=(0, 8))
        col_ic = TEXT_SECONDARY
        ic.create_rectangle(4, 6, 28, 24, outline=col_ic, width=1.8, fill="")
        ic.create_oval(8, 9, 14, 15, outline=col_ic, width=1.2)
        ic.create_line(4, 20, 12, 14, 18, 18, 24, 12, 28, 16, fill=col_ic, width=1.2, smooth=True)
        tk.Label(
            title_frame, text="My Media",
            font=("SF Pro Display", 28, "bold"),
            fg=TEXT_PRIMARY, bg=BG_MAIN
        ).pack(side="left")

        self.count_label = tk.Label(
            header, text="0 photo(s)", font=FONT_SMALL, fg=TEXT_SECONDARY, bg=BG_MAIN
        )
        self.count_label.pack(side="right")

        tk.Label(
            container,
            text="Import your own images (converted to 64×64 grayscale)",
            font=FONT_SUBTITLE, fg=TEXT_SECONDARY, bg=BG_MAIN, anchor="w"
        ).pack(fill="x", pady=(4, 16))

        tk.Frame(container, height=1, bg=SEPARATOR).pack(fill="x")

        action_bar = tk.Frame(container, bg=BG_MAIN)
        action_bar.pack(fill="x", pady=(12, 8))

        GlassButton(action_bar, text="+ Import", command=self._import,
                    width=140, height=36, is_accent=True).pack(side="left")
        GlassButton(action_bar, text="Delete Selected", command=self._delete,
                    width=180, height=36).pack(side="left", padx=(10, 0))

        self.grid_frame = tk.Frame(container, bg=BG_MAIN)
        self.grid_frame.pack(fill="both", expand=True, pady=8)

        self.cards = []
        self.selected_indices = []
        self._refresh()

    def _refresh(self):
        for w in self.grid_frame.winfo_children():
            w.destroy()
        self.cards.clear()
        self.selected_indices.clear()

        photos = self.controller.custom_photos
        self.count_label.config(text=f"{len(photos)} photo(s)")

        if not photos:
            tk.Label(
                self.grid_frame,
                text='No photos imported.\nClick "+ Import" to add images.',
                font=FONT_SUBTITLE, fg=TEXT_SECONDARY, bg=BG_MAIN, justify="center"
            ).place(relx=0.5, rely=0.4, anchor="center")
            return

        for i, img in enumerate(photos):
            card = ImageCard(self.grid_frame, img, i, self._on_card_click, size=100)
            card.grid(row=i // 6, column=i % 6, padx=8, pady=8)
            self.cards.append(card)

    def _on_card_click(self, index, selected):
        if selected and index not in self.selected_indices:
            self.selected_indices.append(index)
        elif not selected and index in self.selected_indices:
            self.selected_indices.remove(index)

    def _import(self):
        files = filedialog.askopenfilenames(
            title="Import Images",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tiff *.webp")]
        )
        if not files:
            return

        for path in files:
            try:
                img = Image.open(path).convert("L")
                dialog = CropDialog(self, img)
                if dialog.result_img:
                    scaled_img = dialog.result_img.resize((64, 64), Image.LANCZOS)
                    arr = np.array(scaled_img, dtype=np.float64) / 255.0
                    self.controller.custom_photos.append(arr)
            except Exception as e:
                messagebox.showerror("Error", f"Could not load:\n{path}\n{e}")

        self._refresh()

    def _delete(self):
        if not self.selected_indices:
            messagebox.showwarning("Selection", "Select photos to delete.")
            return

        for idx in sorted(self.selected_indices, reverse=True):
            if 0 <= idx < len(self.controller.custom_photos):
                self.controller.custom_photos.pop(idx)

        self._refresh()


# =============================================================================
# Grid Screen (Selection + Results)
# =============================================================================

class GridScreen(tk.Frame):
    """Image grid for browsing, selecting, and fusing faces.

    Displays dataset and user images, with per-image weight/mirror
    controls and semantic attribute checkboxes.
    """

    def __init__(self, parent, controller, images, level=1, custom_photo_count=0):
        super().__init__(parent, bg=BG_MAIN)
        self.controller = controller
        self.images = images
        self.cards = []
        self.selected_indices = []
        self.custom_photo_count = custom_photo_count

        header = tk.Frame(self, bg=BG_MAIN)
        header.pack(fill="x", padx=36, pady=(28, 0))

        tk.Label(
            header, text=f"Step {level}",
            font=FONT_TITLE, fg=TEXT_PRIMARY, bg=BG_MAIN
        ).pack(side="left")

        self.counter_label = tk.Label(
            header, text="0 selected", font=FONT_SMALL, fg=TEXT_SECONDARY, bg=BG_MAIN
        )
        self.counter_label.pack(side="right")

        tk.Label(
            self, text="Select images then fuse or save",
            font=FONT_SUBTITLE, fg=TEXT_SECONDARY, bg=BG_MAIN, anchor="w"
        ).pack(fill="x", padx=36, pady=(4, 8))

        # ── Attribute checkboxes ─────────────────────────────────────────
        attr_bar = tk.Frame(self, bg=BG_MAIN)
        attr_bar.pack(fill="x", padx=36, pady=(0, 8))

        attr_labels = {"Bald": "Bald", "Eyeglasses": "Eyeglasses", "Smiling": "Smiling", "Mustache": "Mustache"}
        for attr_key, label_text in attr_labels.items():
            if attr_key in controller.active_attributes:
                tk.Checkbutton(
                    attr_bar, text=label_text,
                    variable=controller.active_attributes[attr_key],
                    font=FONT_SMALL, fg=TEXT_PRIMARY, bg=BG_MAIN,
                    selectcolor=BG_MAIN, activebackground=BG_MAIN
                ).pack(side="left", padx=(0, 12))

        tk.Frame(self, height=1, bg=SEPARATOR).pack(fill="x", padx=36)

        # ── Scrollable area with visible scrollbar ───────────────────────
        scroll_container = tk.Frame(self, bg=BG_MAIN)
        scroll_container.pack(fill="both", expand=True, pady=(8, 0), padx=36)

        self.scrollbar = tk.Scrollbar(scroll_container, orient="vertical")
        self.scrollbar.pack(side="right", fill="y")

        self.scroll_canvas = tk.Canvas(
            scroll_container, bg=BG_MAIN, highlightthickness=0,
            yscrollcommand=self.scrollbar.set
        )
        self.scroll_canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.config(command=self.scroll_canvas.yview)

        self.grid_frame = tk.Frame(self.scroll_canvas, bg=BG_MAIN)
        self.grid_frame.bind("<Configure>",
            lambda e: self.scroll_canvas.configure(scrollregion=self.scroll_canvas.bbox("all")))
        self.scroll_canvas.create_window((0, 0), window=self.grid_frame, anchor="nw")

        # ── Action bar ───────────────────────────────────────────────────
        action_bar = tk.Frame(self, bg=BG_MAIN)
        action_bar.pack(fill="x", padx=36, pady=(0, 24))

        GlassButton(action_bar, text="← Back", command=self._back,
                    width=120, height=38).pack(side="left")
        GlassButton(action_bar, text="+ Import", command=self._open_media,
                    width=120, height=38).pack(side="left", padx=(8, 0))
        GlassButton(action_bar, text="Save", command=self._save,
                    width=150, height=38).pack(side="right", padx=(8, 0))
        GlassButton(action_bar, text="Fuse →", command=self._fuse,
                    width=150, height=38, is_accent=True).pack(side="right")

    def on_show(self):
        self.selected_indices.clear()
        self._build_grid()
        self._update_counter()

    def _build_grid(self):
        for w in self.grid_frame.winfo_children():
            w.destroy()
        self.cards.clear()

        row = 0

        tk.Label(
            self.grid_frame, text="My Photos",
            font=("SF Pro Display", 14, "bold"),
            fg=TEXT_PRIMARY, bg=BG_MAIN, anchor="w"
        ).grid(row=row, column=0, columnspan=GRID_COLUMNS,
               sticky="w", padx=10, pady=(4, 4))
        row += 1

        if self.custom_photo_count > 0:
            for i in range(self.custom_photo_count):
                card = ImageCard(self.grid_frame, self.images[i], i, self._on_card_click)
                card.grid(row=row + i // GRID_COLUMNS, column=i % GRID_COLUMNS,
                          padx=10, pady=10)
                self.cards.append(card)
            row += (self.custom_photo_count - 1) // GRID_COLUMNS + 1
        else:
            tk.Label(
                self.grid_frame, text="No custom photos uploaded. Use '+ Import' below.",
                font=FONT_SMALL, fg=TEXT_SECONDARY, bg=BG_MAIN, anchor="w"
            ).grid(row=row, column=0, columnspan=GRID_COLUMNS, sticky="w", padx=10, pady=10)
            row += 1

        tk.Frame(self.grid_frame, height=1, bg=SEPARATOR).grid(
            row=row, column=0, columnspan=GRID_COLUMNS,
            sticky="ew", padx=10, pady=8)
        row += 1

        tk.Label(
            self.grid_frame, text="Dataset",
            font=("SF Pro Display", 14, "bold"),
            fg=TEXT_PRIMARY, bg=BG_MAIN, anchor="w"
        ).grid(row=row, column=0, columnspan=GRID_COLUMNS,
               sticky="w", padx=10, pady=(4, 4))
        row += 1

        start = self.custom_photo_count
        for j, img_idx in enumerate(range(start, len(self.images))):
            card = ImageCard(self.grid_frame, self.images[img_idx], img_idx, self._on_card_click)
            card.grid(row=row + j // GRID_COLUMNS, column=j % GRID_COLUMNS,
                      padx=10, pady=10)
            self.cards.append(card)

    def _on_card_click(self, index, selected):
        if selected and index not in self.selected_indices:
            self.selected_indices.append(index)
        elif not selected and index in self.selected_indices:
            self.selected_indices.remove(index)
        self._update_counter()

    def _update_counter(self):
        n = len(self.selected_indices)
        self.counter_label.config(text=f"{n} selected")

    def _fuse(self):
        if len(self.selected_indices) < 1:
            messagebox.showwarning("Selection", "Select at least one image.")
            return

        # Collect weights and mirrors from cards
        weights = {}
        mirrors = {}
        for card in self.cards:
            if card.index in self.selected_indices:
                weights[card.index] = card.weight
                mirrors[card.index] = card.is_mirrored

        self.controller.run_fusion(self.images, self.selected_indices, weights=weights, mirrors=mirrors)

    def _save(self):
        self.controller.save_images(self.images, self.selected_indices)

    def _back(self):
        self.controller.go_back()

    def _open_media(self):
        """Import images directly into the active grid and refresh."""
        files = filedialog.askopenfilenames(
            title="Import Images to Fusion",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tiff *.webp")]
        )
        if not files:
            return

        added_count = 0
        for path in files:
            try:
                img = Image.open(path).convert("L")
                dialog = CropDialog(self, img)
                if dialog.result_img:
                    scaled_img = dialog.result_img.resize((64, 64), Image.LANCZOS)
                    arr = np.array(scaled_img, dtype=np.float64) / 255.0
                    
                    # Append to global storage
                    self.controller.custom_photos.append(arr)
                    # Insert securely into active grid representation
                    self.images.insert(self.custom_photo_count, arr)
                    self.custom_photo_count += 1
                    added_count += 1
            except Exception as e:
                messagebox.showerror("Error", f"Could not load:\n{path}\n{e}")

        if added_count > 0:
            # Shift selected_indices for images shifted right
            new_selection = []
            for idx in self.selected_indices:
                if idx >= self.custom_photo_count - added_count:
                    new_selection.append(idx + added_count)
                else:
                    new_selection.append(idx)
            self.selected_indices = new_selection
            
            # Entirely rebuild the current selection
            self._build_grid()
            
            # Since previously selected UI state has been discarded, we must explicitly restore visual borders on the new cards
            for card in self.cards:
                if card.index in self.selected_indices:
                    card.is_selected = True
                    card._update_style()
                    
            self._update_counter()


# =============================================================================
# Credits Screen
# =============================================================================

class CreditsScreen(tk.Frame):
    """Credits page showing authors, contributions, and legal information."""

    def __init__(self, parent, controller):
        super().__init__(parent, bg=BG_MAIN)
        self.controller = controller

        self.scroll_canvas = tk.Canvas(self, bg=BG_MAIN, highlightthickness=0)
        scrollbar = tk.Scrollbar(self, orient="vertical", command=self.scroll_canvas.yview)
        content = tk.Frame(self.scroll_canvas, bg=BG_MAIN)

        content.bind("<Configure>", lambda e: self.scroll_canvas.configure(scrollregion=self.scroll_canvas.bbox("all")))
        self.scroll_canvas.create_window((0, 0), window=content, anchor="nw")
        self.scroll_canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side="right", fill="y")
        self.scroll_canvas.pack(side="left", fill="both", expand=True)

        pad = 48

        # Title
        title_frame = tk.Frame(content, bg=BG_MAIN)
        title_frame.pack(fill="x", padx=pad, pady=(36, 4))
        tk.Label(
            title_frame, text="Credits",
            font=("SF Pro Display", 28, "bold"),
            fg=TEXT_PRIMARY, bg=BG_MAIN
        ).pack(side="left")

        tk.Label(
            content, text="Authors, contributions, and legal notices",
            font=FONT_SUBTITLE, fg=TEXT_SECONDARY, bg=BG_MAIN, anchor="w"
        ).pack(fill="x", padx=pad, pady=(0, 20))

        tk.Frame(content, height=1, bg=SEPARATOR).pack(fill="x", padx=pad)

        # Version
        version = self._get_version()
        tk.Label(
            content, text=f"Version {version}",
            font=("SF Pro Text", 12, "bold"), fg=ACCENT, bg=BG_MAIN, anchor="w"
        ).pack(fill="x", padx=pad, pady=(20, 12))

        sections = [
            ("Project",
             "ProfileX — Face generation using a Variational Autoencoder\n"
             "and genetic algorithm. Built with Python, PyTorch, and Tkinter."),
            ("Authors",
             "Nathanaël Pons, Axel Mondesir, Ambre Lesurtel, Skander Zouhir, Eglantine Salomé.\n"
             "Developed as part of a BIM engineering project."),
            ("Technology",
             "• VAE trained on CelebA (202,599 celebrity faces)\n"
             "• Genetic algorithm for latent-space mutation\n"
             "• Semantic attribute manipulation (Bald, Eyeglasses, Smiling, Mustache)\n"
             "• Weighted fusion with image mirroring support"),
            ("Datasets",
             "CelebA — Large-scale CelebFaces Attributes Dataset\n"
             "Citation: Liu, Z., Luo, P., Wang, X., & Tang, X. (2015).\n"
             "Deep Learning Face Attributes in the Wild. In ICCV."),
            ("License",
             "This software is provided for academic and educational purposes.\n"
             "All rights reserved by the project authors."),
        ]

        for title, body in sections:
            tk.Label(
                content, text=title, font=("SF Pro Display", 18, "bold"),
                fg=TEXT_PRIMARY, bg=BG_MAIN, anchor="w"
            ).pack(fill="x", padx=pad, pady=(24, 6))
            tk.Label(
                content, text=body, font=FONT_SUBTITLE, fg=TEXT_SECONDARY,
                bg=BG_MAIN, anchor="w", justify="left"
            ).pack(fill="x", padx=(pad + 8, pad))

        tk.Frame(content, height=40, bg=BG_MAIN).pack()

    @staticmethod
    def _get_version():
        """Retrieve the current release version of ProfileX."""
        return "1.0.0"
