import math
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image
import numpy as np

from .constants import *
from .widgets import GlassButton, ImageCard


# =============================================================================
# Home Screen
# =============================================================================

class HomeScreen(tk.Frame):
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
            container, text="Image Fusion",
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
    """Settings panel embedded in a tab."""

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
    """Documentation page embedded in a tab."""

    def __init__(self, parent, controller):
        super().__init__(parent, bg=BG_MAIN)
        self.controller = controller

        canvas = tk.Canvas(self, bg=BG_MAIN, highlightthickness=0)
        scrollbar = tk.Scrollbar(self, orient="vertical", command=canvas.yview)
        content = tk.Frame(canvas, bg=BG_MAIN)

        content.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=content, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        # Scroll binding (macOS-compatible)
        def _on_scroll(event):
            canvas.yview_scroll(-event.delta, "units")
        canvas.bind("<MouseWheel>", _on_scroll)
        content.bind("<MouseWheel>", _on_scroll)

        def _enter(event):
            canvas.bind_all("<MouseWheel>", _on_scroll)
        def _leave(event):
            canvas.unbind_all("<MouseWheel>")
        canvas.bind("<Enter>", _enter)
        canvas.bind("<Leave>", _leave)

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
            title_frame, text="Documentation",
            font=("SF Pro Display", 28, "bold"),
            fg=TEXT_PRIMARY, bg=BG_MAIN
        ).pack(side="left")

        tk.Label(
            content, text="User guide for the image fusion application",
            font=FONT_SUBTITLE, fg=TEXT_SECONDARY, bg=BG_MAIN, anchor="w"
        ).pack(fill="x", padx=pad, pady=(0, 20))

        tk.Frame(content, height=1, bg=SEPARATOR).pack(fill="x", padx=pad)

        sections = [
            ("How does it work?",
             "The application uses a convolutional autoencoder trained on the\n"
             "Olivetti Faces dataset. Each face is encoded into a low-dimensional\n"
             "latent space, then latent vectors are combined by fusion (average)\n"
             "and mutation (genetic algorithm) to generate new faces."),
            ("Step 1 — Selection",
             "Click 'Get Started' to load a sample of faces.\n"
             "Select one or more faces by clicking on them.\n"
             "A red border indicates selected images."),
            ("Step 2 — Fusion",
             "Click 'Fuse →' to start the process.\n"
             "• 1 selected image → generates variants by mutation only.\n"
             "• 2+ images → fusion of latent vectors + mutation of results.\n"
             "The number of variants and mutation parameters are\n"
             "configurable in the Settings tab."),
            ("Step 3 — Save",
             "Select the desired results and click 'Save'.\n"
             "Images are exported as PNG + ZIP archive in the chosen folder."),
            ("Settings",
             "• Variants: number of mutations generated (1 to 20).\n"
             "• Mutation Rate: probability that a gene is mutated (0 to 1).\n"
             "• Mutation Level: amplitude of mutation noise (1 to 10)."),
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


# =============================================================================
# Media Library Screen
# =============================================================================

class MediaScreen(tk.Frame):
    """Custom photo library management."""

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
                img = Image.open(path).convert("L").resize((64, 64), Image.LANCZOS)
                arr = np.array(img, dtype=np.float64) / 255.0
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
    """Generic screen displaying an image grid with Back, Fuse, and Save."""

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
        ).pack(fill="x", padx=36, pady=(4, 16))

        tk.Frame(self, height=1, bg=SEPARATOR).pack(fill="x", padx=36)

        # Scrollable area
        self.scroll_canvas = tk.Canvas(self, bg=BG_MAIN, highlightthickness=0)
        self.grid_frame = tk.Frame(self.scroll_canvas, bg=BG_MAIN)
        self.grid_frame.bind("<Configure>",
            lambda e: self.scroll_canvas.configure(scrollregion=self.scroll_canvas.bbox("all")))
        self.scroll_canvas.create_window((0, 0), window=self.grid_frame, anchor="nw")
        self.scroll_canvas.pack(fill="both", expand=True, pady=(8, 0), padx=36)

        # Scroll macOS-compatible
        def _on_scroll(event):
            self.scroll_canvas.yview_scroll(-event.delta, "units")
        self.scroll_canvas.bind("<MouseWheel>", _on_scroll)
        self.grid_frame.bind("<MouseWheel>", _on_scroll)

        def _enter(event):
            self.scroll_canvas.bind_all("<MouseWheel>", _on_scroll)
        def _leave(event):
            self.scroll_canvas.unbind_all("<MouseWheel>")
        self.scroll_canvas.bind("<Enter>", _enter)
        self.scroll_canvas.bind("<Leave>", _leave)

        action_bar = tk.Frame(self, bg=BG_MAIN)
        action_bar.pack(fill="x", padx=36, pady=(0, 24))

        GlassButton(action_bar, text="← Back", command=self._back,
                    width=120, height=38).pack(side="left")
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

        if self.custom_photo_count > 0:
            tk.Label(
                self.grid_frame, text="My Photos",
                font=("SF Pro Display", 14, "bold"),
                fg=TEXT_PRIMARY, bg=BG_MAIN, anchor="w"
            ).grid(row=row, column=0, columnspan=GRID_COLUMNS,
                   sticky="w", padx=10, pady=(4, 4))
            row += 1

            for i in range(self.custom_photo_count):
                card = ImageCard(self.grid_frame, self.images[i], i, self._on_card_click)
                card.grid(row=row + i // GRID_COLUMNS, column=i % GRID_COLUMNS,
                          padx=10, pady=10)
                self.cards.append(card)
            row += (self.custom_photo_count - 1) // GRID_COLUMNS + 1

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
        self.controller.run_fusion(self.images, self.selected_indices)

    def _save(self):
        self.controller.save_images(self.images, self.selected_indices)

    def _back(self):
        self.controller.go_back()
