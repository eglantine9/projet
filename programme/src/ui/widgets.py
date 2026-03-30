import tkinter as tk
from .constants import *
from .utils import numpy_to_photo


class TabBar(tk.Frame):
    """Horizontal browser-style tab bar."""

    def __init__(self, parent, controller):
        super().__init__(parent, bg=TAB_BAR_BG, height=TAB_HEIGHT)
        self.controller = controller
        self.pack_propagate(False)

        self.tab_frame = tk.Frame(self, bg=TAB_BAR_BG)
        self.tab_frame.pack(side="left", fill="y", padx=(4, 0))

        self.btn_new = tk.Label(
            self, text="+", font=("SF Pro Text", 18),
            fg=TEXT_SECONDARY, bg=TAB_BAR_BG, cursor="hand2", padx=8
        )
        self.btn_new.pack(side="left", fill="y")
        self.btn_new.bind("<Button-1>", lambda e: controller.new_fusion_tab())
        self.btn_new.bind("<Enter>", lambda e: self.btn_new.config(fg=TEXT_PRIMARY))
        self.btn_new.bind("<Leave>", lambda e: self.btn_new.config(fg=TEXT_SECONDARY))

        self.tabs = []
        self.active_idx = -1

    def add(self, title, closable=True):
        idx = len(self.tabs)
        frame = tk.Frame(self.tab_frame, bg=TAB_INACTIVE_BG, cursor="hand2")
        frame.pack(side="left", fill="y", padx=(0, 1), pady=(4, 0))

        label = tk.Label(
            frame, text=title, font=FONT_SMALL,
            fg=TEXT_SECONDARY, bg=TAB_INACTIVE_BG, padx=12, pady=6
        )
        label.pack(side="left")

        btn_close = None
        if closable:
            btn_close = tk.Label(
                frame, text="×", font=("SF Pro Text", 13),
                fg=TEXT_SECONDARY, bg=TAB_INACTIVE_BG, cursor="hand2", padx=4
            )
            btn_close.pack(side="left", pady=6)
            btn_close.bind("<Button-1>", lambda e, i=idx: self._on_close(i))
            btn_close.bind("<Enter>", lambda e, w=btn_close: w.config(fg="#FF3B30"))
            btn_close.bind("<Leave>", lambda e, i=idx: self._reset_close_style(i))

        info = {"frame": frame, "label": label, "btn_close": btn_close}
        self.tabs.append(info)

        for w in [frame, label]:
            w.bind("<Button-1>", lambda e, i=idx: self.controller.select_tab(i))

        return idx

    def _on_close(self, idx):
        self.controller.close_tab(idx)

    def _reset_close_style(self, idx):
        if 0 <= idx < len(self.tabs) and self.tabs[idx]["btn_close"]:
            bg = TAB_ACTIVE_BG if idx == self.active_idx else TAB_INACTIVE_BG
            self.tabs[idx]["btn_close"].config(fg=TEXT_SECONDARY, bg=bg)

    def activate(self, idx):
        self.active_idx = idx
        for i, tab in enumerate(self.tabs):
            is_active = (i == idx)
            bg = TAB_ACTIVE_BG if is_active else TAB_INACTIVE_BG
            fg = TEXT_PRIMARY if is_active else TEXT_SECONDARY
            tab["frame"].config(bg=bg)
            tab["label"].config(bg=bg, fg=fg)
            if tab["btn_close"]:
                tab["btn_close"].config(bg=bg)

    def remove(self, idx):
        self.tabs[idx]["frame"].destroy()
        self.tabs.pop(idx)
        self._rebind_all()

    def rename(self, idx, title):
        if 0 <= idx < len(self.tabs):
            self.tabs[idx]["label"].config(text=title)

    def _rebind_all(self):
        for i, tab in enumerate(self.tabs):
            for w in [tab["frame"], tab["label"]]:
                w.bind("<Button-1>", lambda e, ii=i: self.controller.select_tab(ii))
            if tab["btn_close"]:
                tab["btn_close"].bind("<Button-1>", lambda e, ii=i: self._on_close(ii))
                tab["btn_close"].bind("<Leave>", lambda e, ii=i: self._reset_close_style(ii))


class GlassButton(tk.Canvas):
    """Rounded glass-effect button."""

    def __init__(self, parent, text, command, width=200, height=42,
                 bg_color=BTN_GLASS, text_color=TEXT_PRIMARY, is_accent=False):
        super().__init__(
            parent, width=width, height=height,
            bg=parent["bg"], highlightthickness=0, cursor="hand2"
        )
        self.command = command
        self.text = text
        self._width = width
        self._height = height
        self.base_color = ACCENT if is_accent else bg_color
        self.hover_color = ACCENT_HOVER if is_accent else BTN_GLASS_HOVER
        self.text_color = "#ffffff" if is_accent else text_color

        self._draw(self.base_color)

        self.bind("<Enter>", lambda e: self._draw(self.hover_color))
        self.bind("<Leave>", lambda e: self._draw(self.base_color))
        self.bind("<Button-1>", lambda e: self.command())

    def _draw(self, color):
        self.delete("all")
        r = 10
        w, h = self._width, self._height
        self.create_arc(0, 0, r * 2, r * 2, start=90, extent=90, fill=color, outline="")
        self.create_arc(w - r * 2, 0, w, r * 2, start=0, extent=90, fill=color, outline="")
        self.create_arc(0, h - r * 2, r * 2, h, start=180, extent=90, fill=color, outline="")
        self.create_arc(w - r * 2, h - r * 2, w, h, start=270, extent=90, fill=color, outline="")
        self.create_rectangle(r, 0, w - r, h, fill=color, outline="")
        self.create_rectangle(0, r, w, h - r, fill=color, outline="")
        self.create_text(w / 2, h / 2, text=self.text, fill=self.text_color, font=FONT_BUTTON)


class ImageCard(tk.Frame):
    """Thumbnail with rounded corners, shadow, and selection border."""

    def __init__(self, parent, image_np, index, callback, size=THUMBNAIL_SIZE):
        super().__init__(parent, bg=parent["bg"])
        self.index = index
        self.is_selected = False
        self.callback = callback
        self.size = size

        self.shadow_frame = tk.Frame(self, bg=SHADOW, padx=1, pady=1)
        self.shadow_frame.pack()

        self.card_frame = tk.Frame(self.shadow_frame, bg=CARD_BG, padx=3, pady=3)
        self.card_frame.pack()

        self.photo = numpy_to_photo(image_np, size)
        self.image_label = tk.Label(
            self.card_frame, image=self.photo, bg=CARD_BG, cursor="hand2"
        )
        self.image_label.pack()
        self.image_label.bind("<Button-1>", self._on_click)

    def _on_click(self, event):
        self.is_selected = not self.is_selected
        self._update_style()
        self.callback(self.index, self.is_selected)

    def _update_style(self):
        if self.is_selected:
            self.shadow_frame.config(bg=SELECTION_BORDER)
            self.card_frame.config(bg=SELECTION_BORDER)
        else:
            self.shadow_frame.config(bg=SHADOW)
            self.card_frame.config(bg=CARD_BG)
