import math
import os
import zipfile
import tkinter as tk
from tkinter import filedialog, messagebox
from datetime import datetime
from PIL import Image

import numpy as np
import torch
from sklearn.datasets import fetch_olivetti_faces

# Core imports
from ..core.logic import load_trained_model, encode_images, decode_latents, fuse_parents, mutate_vector

# UI imports
from .constants import *
from .widgets import TabBar
from .screens import HomeScreen, SettingsScreen, DocsScreen, MediaScreen, GridScreen

# Determine asset path relative to this file
_UI_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_UI_DIR)
_MODEL_PATH = os.path.join(_SRC_DIR, "assets", "model.pth")

# Global model instance
model, latent_dim = load_trained_model(_MODEL_PATH)


class Application(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Image Fusion")
        self.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        self.configure(bg=BG_MAIN)
        self.resizable(False, False)

        self.variant_count = DEFAULT_VARIANT_COUNT
        self.mutation_rate = DEFAULT_MUTATION_RATE
        self.mutation_level = DEFAULT_MUTATION_LEVEL

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

        self._add_sidebar_button(bar, self._draw_docs_icon, "Docs", self._open_docs)
        self._add_sidebar_button(bar, self._draw_media_icon, "Media", self._open_media)

        tk.Frame(bar, bg=BG_SIDEBAR).pack(fill="y", expand=True)
        tk.Frame(bar, height=1, bg="#3a3a3c").pack(fill="x", padx=10, pady=5)

        self._add_sidebar_button(bar, self._draw_settings_icon, "Settings", self._open_settings)

        tk.Frame(bar, height=15, bg=BG_SIDEBAR).pack()

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
        data = fetch_olivetti_faces(shuffle=True, random_state=42)
        indices = np.random.RandomState(0).choice(len(data.images), size=10, replace=False)
        dataset_images = [data.images[i] for i in indices]
        custom_count = len(self.custom_photos)
        images = list(self.custom_photos) + dataset_images
        self.push_screen(images, custom_photo_count=custom_count)

    def run_fusion(self, source_images, selected_indices):
        selected = [source_images[k] for k in selected_indices]
        if len(selected_indices) == 1:
            images_t = torch.tensor(np.array(selected), dtype=torch.float32).unsqueeze(1)
            latent = encode_images(model, images_t)
            mutated = mutate_vector(latent, self.variant_count + 1, self.mutation_rate, self.mutation_level)
            results = decode_latents(model, mutated).squeeze(1)
        else:
            images_t = torch.tensor(np.array(selected), dtype=torch.float32).unsqueeze(1)
            latent = encode_images(model, images_t)
            fused = fuse_parents(latent)
            fused_result = decode_latents(model, fused).squeeze(1)
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
        self._open_unique_tab("docs", "Documentation", DocsScreen)

    def _open_settings(self):
        self._open_unique_tab("settings", "Settings", SettingsScreen)

    def _open_media(self):
        self._open_unique_tab("media", "My Media", MediaScreen)
