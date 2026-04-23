import math
import os
import zipfile
import tkinter as tk
from tkinter import filedialog, messagebox
from datetime import datetime
from PIL import Image

import numpy as np
import torch


# Core imports
from ..core.logic import load_trained_model, encode_images, decode_latents, fuse_weighted, load_attribute_directions
from ..core.genetic_algorithm import fuse_parents, mutate_vector

# UI imports
from .constants import *
from .widgets import TabBar
from .screens import HomeScreen, SettingsScreen, DocsScreen, MediaScreen, GridScreen, CreditsScreen

import sys

# Determine path resolution for normal execution vs PyInstaller
if getattr(sys, 'frozen', False):
    # PyInstaller extracts code to sys._MEIPASS (Contents/Frameworks in .app)
    # but BUNDLE copies datas to Contents/Resources — check both locations.
    _MEIPASS = sys._MEIPASS
    _RESOURCES = os.path.join(os.path.dirname(_MEIPASS), "Resources")
    # Prefer Resources (where BUNDLE puts datas), fall back to _MEIPASS
    _BUNDLE_DIR = _RESOURCES if os.path.isdir(_RESOURCES) else _MEIPASS
else:
    _UI_DIR = os.path.dirname(os.path.abspath(__file__))
    _SRC_DIR = os.path.dirname(_UI_DIR)
    _BUNDLE_DIR = os.path.dirname(os.path.dirname(_SRC_DIR))

_MODEL_PATH = os.path.join(_BUNDLE_DIR, "programme", "src", "assets", "vae_celeba_final.pth")
_ATTR_DIR_PATH = os.path.join(_BUNDLE_DIR, "data", "attribute_directions.npz")
_ATTR_VEC_PATH = os.path.join(_BUNDLE_DIR, "data", "attribute_vectors.npz")

# Global model instance
model, latent_dim = load_trained_model(_MODEL_PATH)

# Attribute directions for semantic control
try:
    attribute_directions = load_attribute_directions(_ATTR_DIR_PATH)
except Exception:
    attribute_directions = {}

# Attribute mean vectors (used for Bald which works better with mean vectors)
try:
    _attr_vec_data = np.load(_ATTR_VEC_PATH)
    attribute_vectors = {k: _attr_vec_data[k] for k in _attr_vec_data.files}
except Exception:
    attribute_vectors = {}


class Application(tk.Tk):
    """Main application window managing tabs, sidebar, and fusion pipeline."""

    def __init__(self):
        super().__init__()

        self.title("Image Fusion")
        self._setup_icon()
        self.bind_all("<MouseWheel>", self._on_mousewheel)
        self.bind_all("<Button-4>", self._on_mousewheel)
        self.bind_all("<Button-5>", self._on_mousewheel)
        self.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        self.configure(bg=BG_MAIN)
        self.resizable(True, True)
        self.minsize(900, 600)

        self.variant_count = DEFAULT_VARIANT_COUNT
        self.mutation_rate = DEFAULT_MUTATION_RATE
        self.mutation_level = DEFAULT_MUTATION_LEVEL

        # Semantic attributes
        self.active_attributes = {
            "Bald": tk.BooleanVar(value=False),
            "Eyeglasses": tk.BooleanVar(value=False),
            "Smiling": tk.BooleanVar(value=False),
            "Mustache": tk.BooleanVar(value=False),
        }
        self.attribute_scale = 3.0  # ~10% of typical latent norm (~29)

        self.root_frame = tk.Frame(self, bg=BG_MAIN)
        self.root_frame.pack(fill="both", expand=True)

        self._create_sidebar()

        # Right zone: tabs + content
        self.right_frame = tk.Frame(self.root_frame, bg=BG_MAIN)
        self.right_frame.pack(side="left", fill="both", expand=True)

        self.tab_bar = TabBar(self.right_frame, self)
        self.tab_bar.pack(fill="x")

        self.tab_container = tk.Frame(self.right_frame, bg=BG_MAIN)
        self.tab_container.pack(fill="both", expand=True)

        # Tab data
        self.tabs = []
        self.active_tab_idx = -1
        self.custom_photos = []

        self.new_fusion_tab()

        # macOS: Cmd+W triggers ::tk::mac::CloseWindow internally
        self.createcommand("::tk::mac::CloseWindow", self._on_close_shortcut)
        self.protocol("WM_DELETE_WINDOW", self._on_close_shortcut)

        # Ctrl+W fallback
        self.bind_all("<Control-w>", lambda e: self._on_close_shortcut())

        # Native menu
        menubar = tk.Menu(self)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Close Tab", accelerator="⌘W", command=self._on_close_shortcut)
        file_menu.add_separator()
        file_menu.add_command(label="Quit", accelerator="⌘Q", command=self.destroy)
        menubar.add_cascade(label="File", menu=file_menu)
        self.config(menu=menubar)

    def _on_close_shortcut(self):
        if len(self.tabs) > 1 and self.active_tab_idx >= 0:
            self.close_tab(self.active_tab_idx)
        elif len(self.tabs) <= 1:
            self.destroy()

    # ── Sidebar ──────────────────────────────────────────────────────────

    def _create_sidebar(self):
        bar = tk.Frame(
            self.root_frame, bg=BG_SIDEBAR,
            width=SIDEBAR_WIDTH, height=WINDOW_HEIGHT
        )
        bar.pack(side="left", fill="y")
        bar.pack_propagate(False)

        tk.Frame(bar, height=20, bg=BG_SIDEBAR).pack()

        logo = tk.Canvas(bar, width=36, height=36, bg=BG_SIDEBAR, highlightthickness=0)
        logo.pack(pady=(10, 30))
        logo.create_oval(4, 4, 32, 32, fill=ACCENT, outline="")
        logo.create_oval(12, 12, 24, 24, fill="#ffffff", outline="")
        logo.create_oval(15, 15, 21, 21, fill=ACCENT, outline="")

        self._add_sidebar_button(bar, self._draw_home_icon, "Home", self._go_home)
        self._add_sidebar_button(bar, self._draw_docs_icon, "Guide", self._open_docs)
        self._add_sidebar_button(bar, self._draw_media_icon, "Media", self._open_media)

        tk.Frame(bar, bg=BG_SIDEBAR).pack(fill="y", expand=True)
        tk.Frame(bar, height=1, bg="#3a3a3c").pack(fill="x", padx=10, pady=5)

        self._add_sidebar_button(bar, self._draw_credits_icon, "Credits", self._open_credits)
        self._add_sidebar_button(bar, self._draw_settings_icon, "Settings", self._open_settings)

        tk.Frame(bar, height=15, bg=BG_SIDEBAR).pack()

    @staticmethod
    def _draw_home_icon(canvas):
        """Draw a house icon on the given canvas."""
        c, col = canvas, TEXT_SIDEBAR
        c.create_polygon(20, 6, 8, 16, 12, 16, 12, 26, 28, 26, 28, 16, 32, 16,
                         fill="", outline=col, width=1.5)
        c.create_rectangle(17, 18, 23, 26, outline=col, width=1.2, fill="")

    @staticmethod
    def _draw_credits_icon(canvas):
        """Draw an info-circle icon on the given canvas."""
        c, col = canvas, TEXT_SIDEBAR
        c.create_oval(8, 4, 32, 28, outline=col, width=1.5, fill="")
        c.create_text(20, 10, text="i", font=("SF Pro Text", 12, "bold"), fill=col)
        c.create_line(20, 16, 20, 24, fill=col, width=1.5)

    @staticmethod
    def _draw_docs_icon(canvas):
        c, col = canvas, TEXT_SIDEBAR
        c.create_polygon(10, 6, 20, 8, 20, 26, 10, 24, fill="", outline=col, width=1.5)
        c.create_polygon(20, 8, 30, 6, 30, 24, 20, 26, fill="", outline=col, width=1.5)
        for y in (12, 16, 20):
            c.create_line(13, y, 18, y + 0.5, fill=col, width=1)
            c.create_line(22, y + 0.5, 27, y, fill=col, width=1)

    @staticmethod
    def _draw_media_icon(canvas):
        c, col = canvas, TEXT_SIDEBAR
        c.create_rectangle(8, 8, 32, 26, outline=col, width=1.5, fill="")
        c.create_oval(12, 11, 18, 17, outline=col, width=1)
        c.create_line(8, 22, 16, 16, 22, 20, 28, 14, 32, 18, fill=col, width=1.2, smooth=True)

    @staticmethod
    def _draw_settings_icon(canvas):
        c, col = canvas, TEXT_SIDEBAR
        cx, cy, r_ext, r_int = 20, 16, 11, 7
        points = []
        for i in range(16):
            angle = math.pi * i / 8 - math.pi / 2
            r = r_ext if i % 2 == 0 else r_int
            points.extend([cx + r * math.cos(angle), cy + r * math.sin(angle)])
        c.create_polygon(points, fill="", outline=col, width=1.5)
        c.create_oval(cx - 4, cy - 4, cx + 4, cy + 4, fill="", outline=col, width=1.5)

    def _add_sidebar_button(self, parent, draw_icon, text, command):
        frame = tk.Frame(parent, bg=BG_SIDEBAR, cursor="hand2")
        frame.pack(pady=4, padx=6, fill="x")

        icon_canvas = tk.Canvas(frame, width=40, height=32, bg=BG_SIDEBAR, highlightthickness=0)
        icon_canvas.pack(pady=(6, 0))
        draw_icon(icon_canvas)

        label = tk.Label(
            frame, text=text, font=FONT_SIDEBAR, fg=TEXT_SIDEBAR_DIM, bg=BG_SIDEBAR
        )
        label.pack(pady=(0, 6))

        def enter(e):
            frame.config(bg=SIDEBAR_HOVER)
            icon_canvas.config(bg=SIDEBAR_HOVER)
            label.config(bg=SIDEBAR_HOVER)

        def leave(e):
            frame.config(bg=BG_SIDEBAR)
            icon_canvas.config(bg=BG_SIDEBAR)
            label.config(bg=BG_SIDEBAR)

        for widget in (frame, icon_canvas, label):
            widget.bind("<Enter>", enter)
            widget.bind("<Leave>", leave)
            widget.bind("<Button-1>", lambda e, c=command: c())

    # ── Tab Management ───────────────────────────────────────────────────

    def new_fusion_tab(self):
        count = sum(1 for t in self.tabs if t["type"] == "fusion") + 1
        title = f"Fusion {count}"

        frame = tk.Frame(self.tab_container, bg=BG_MAIN)
        frame.place(relwidth=1, relheight=1)

        home_screen = HomeScreen(parent=frame, controller=self)
        home_screen.place(relwidth=1, relheight=1)

        tab = {
            "title": title, "type": "fusion",
            "frame": frame, "screen_stack": [],
            "home_screen": home_screen,
        }
        self.tabs.append(tab)
        idx = len(self.tabs) - 1
        self.tab_bar.add(title, closable=(idx > 0))
        self.select_tab(idx)

    def _open_unique_tab(self, tab_type, title, screen_class):
        """Open or reactivate a singleton tab."""
        for i, t in enumerate(self.tabs):
            if t["type"] == tab_type:
                self.select_tab(i)
                return

        frame = tk.Frame(self.tab_container, bg=BG_MAIN)
        frame.place(relwidth=1, relheight=1)

        screen = screen_class(parent=frame, controller=self)
        screen.place(relwidth=1, relheight=1)

        tab = {
            "title": title, "type": tab_type,
            "frame": frame, "screen_stack": [],
            "home_screen": None,
        }
        self.tabs.append(tab)
        idx = len(self.tabs) - 1
        self.tab_bar.add(title, closable=True)
        self.select_tab(idx)

    def select_tab(self, idx):
        if 0 <= idx < len(self.tabs):
            self.active_tab_idx = idx
            self.tabs[idx]["frame"].tkraise()
            self.tab_bar.activate(idx)

    def close_tab(self, idx):
        if len(self.tabs) <= 1:
            return
        tab = self.tabs[idx]
        for screen in tab["screen_stack"]:
            screen.destroy()
        tab["frame"].destroy()
        self.tabs.pop(idx)
        self.tab_bar.remove(idx)

        if self.active_tab_idx == idx:
            self.active_tab_idx = min(idx, len(self.tabs) - 1)
        elif self.active_tab_idx > idx:
            self.active_tab_idx -= 1
        self.select_tab(self.active_tab_idx)

    # ── In-tab Navigation ────────────────────────────────────────────────

    def push_screen(self, images, custom_photo_count=0):
        tab = self.tabs[self.active_tab_idx]
        level = len(tab["screen_stack"]) + 1
        screen = GridScreen(
            parent=tab["frame"], controller=self,
            images=images, level=level,
            custom_photo_count=custom_photo_count
        )
        screen.place(relwidth=1, relheight=1)
        tab["screen_stack"].append(screen)
        screen.tkraise()
        screen.on_show()
        self.tab_bar.rename(self.active_tab_idx, f"{tab['title']} · Step {level}")

    def go_back(self):
        tab = self.tabs[self.active_tab_idx]
        if tab["screen_stack"]:
            tab["screen_stack"].pop().destroy()
        if tab["screen_stack"]:
            tab["screen_stack"][-1].tkraise()
            self.tab_bar.rename(
                self.active_tab_idx,
                f"{tab['title']} · Step {len(tab['screen_stack'])}"
            )
        elif tab["home_screen"]:
            tab["home_screen"].tkraise()
            self.tab_bar.rename(self.active_tab_idx, tab["title"])



    # ── Business Logic ───────────────────────────────────────────────────

    def load_sample(self):
        celeba_dir = os.path.join(_BUNDLE_DIR, "data", "celeba", "img_align_celeba")
        
        try:
            # We assume images are mostly .jpg in that directory
            files = [f for f in os.listdir(celeba_dir) if f.endswith(".jpg")]
        except FileNotFoundError:
            messagebox.showerror("Error", "CelebA dataset not found. Please ensure it is in the data/celeba directory.")
            return

        if not files:
            messagebox.showerror("Error", "CelebA dataset is empty.")
            return

        selected_files = np.random.choice(files, size=10, replace=False)
        dataset_images = []
        for file in selected_files:
            path = os.path.join(celeba_dir, file)
            img = Image.open(path)
            
            w, h = img.size
            crop_size = 178
            left = (w - crop_size) // 2
            top = (h - crop_size) // 2
            right = (w + crop_size) // 2
            bottom = (h + crop_size) // 2
            
            img = img.crop((left, top, right, bottom))
            img = img.resize((64, 64), Image.LANCZOS)
            img = img.convert("L")
            img_arr = np.array(img, dtype=np.float32) / 255.0
            
            dataset_images.append(img_arr)
            
        custom_count = len(self.custom_photos)
        images = list(self.custom_photos) + dataset_images
        self.push_screen(images, custom_photo_count=custom_count)

    def run_fusion(self, source_images, selected_indices, weights=None, mirrors=None):
        # Prepare images (apply mirror if needed)
        selected = []
        for k in selected_indices:
            img = source_images[k]
            if mirrors and mirrors.get(k, False):
                img = np.fliplr(img).copy()
            selected.append(img)

        images_t = torch.tensor(np.array(selected), dtype=torch.float32).unsqueeze(1)
        latent = encode_images(model, images_t)

        if len(selected_indices) == 1:
            fused = latent.squeeze(0)
        else:
            # Weighted or simple fusion
            if weights:
                w = [weights.get(k, 1.0) for k in selected_indices]
                fused = fuse_weighted(latent, w)
            else:
                fused = fuse_parents(latent)

        # Apply semantic attributes
        # Each entry: (scale, source_dict)
        # Bald uses mean vectors; others use direction vectors
        attr_config = {
            "Bald":       (7.0,  attribute_vectors),
            "Eyeglasses": (10.0, attribute_directions),
            "Smiling":    (10.0, attribute_directions),
            "Mustache":   (10.0, attribute_directions),
        }

        for attr_name, var in self.active_attributes.items():
            if var.get() and attr_name in attr_config:
                scale, source = attr_config[attr_name]
                if attr_name in source:
                    fused = fused + scale * source[attr_name]

        # Decode fused result
        fused_result = decode_latents(model, fused).squeeze(1)

        # Mutate
        mutated = mutate_vector(fused, self.variant_count, self.mutation_rate, self.mutation_level)
        mutated_results = decode_latents(model, mutated).squeeze(1)

        results = torch.cat([fused_result, mutated_results], dim=0)
        result_images = results.detach().cpu().numpy()
        self.push_screen(list(result_images))

    def save_images(self, images, indices):
        if not indices:
            messagebox.showwarning("No Selection", "Select at least one image.")
            return

        folder = filedialog.askdirectory(title="Save Location")
        if not folder:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(folder, f"fusion_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)

        paths = []
        for idx in indices:
            img = (images[idx] * 255).astype(np.uint8)
            pil = Image.fromarray(img, mode="L")
            path = os.path.join(output_dir, f"result_{idx + 1}.png")
            pil.save(path)
            paths.append(path)

        zip_path = os.path.join(folder, f"fusion_{timestamp}.zip")
        with zipfile.ZipFile(zip_path, "w") as zf:
            for p in paths:
                zf.write(p, os.path.basename(p))

        messagebox.showinfo(
            "Save Complete",
            f"{len(paths)} image(s) → {output_dir}\nArchive → {zip_path}"
        )

    # ── Open Special Tabs ────────────────────────────────────────────────

    def _open_docs(self):
        self._open_unique_tab("docs", "Guide", DocsScreen)

    def _open_settings(self):
        self._open_unique_tab("settings", "Settings", SettingsScreen)

    def _open_media(self):
        """Open the My Media tab."""
        self._open_unique_tab("media", "My Media", MediaScreen)

    def _open_credits(self):
        """Open the Credits tab."""
        self._open_unique_tab("credits", "Credits", CreditsScreen)

    def _go_home(self):
        """Navigate to the first tab."""
        if self.tabs:
            self.select_tab(0)

    # ── Global Setup/Events ──────────────────────────────────────────────────

    def _setup_icon(self):
        """Generate and set a custom application icon."""
        try:
            from PIL import Image, ImageDraw, ImageTk
            size = 256
            img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
            d = ImageDraw.Draw(img)
            d.rounded_rectangle([(0, 0), (size, size)], radius=64, fill="#1c1c1e")
            d.ellipse([(64, 64), (192, 192)], fill="#ffffff")
            d.ellipse([(96, 96), (160, 160)], fill="#0a84ff")
            self._icon_img = ImageTk.PhotoImage(img) # Prevent garbage collection
            self.iconphoto(True, self._icon_img)
        except Exception as e:
            print(f"Icon setup failed: {e}")

    def _on_mousewheel(self, event):
        """Global handler to scroll the active tab's Canvas on macOS/Linux."""
        try:
            if self.active_tab_idx < 0:
                return
            tab = self.tabs[self.active_tab_idx]
            
            # Find the foremost screen of the current tab
            if tab.get("type") == "fusion":
                screen = tab["screen_stack"][-1] if tab.get("screen_stack") else tab.get("home_screen")
            else:
                screen = tab.get("current_screen")
                
            if screen and hasattr(screen, "scroll_canvas"):
                raw_delta = getattr(event, "delta", 0)
                
                # Determine standard trackpad velocity
                if raw_delta == 0:
                    if getattr(event, "num", 0) == 4: delta = -1
                    elif getattr(event, "num", 0) == 5: delta = 1
                    else: return
                else:
                    delta = int(-raw_delta)
                    if delta == 0:
                        delta = -1 if raw_delta > 0 else 1
                
                # Apply native scroll directly to the canvas component
                screen.scroll_canvas.yview_scroll(delta, "units")
        except Exception as e:
            print(f"Scroll error: {e}")
