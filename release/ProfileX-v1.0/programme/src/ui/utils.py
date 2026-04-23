import numpy as np
from PIL import Image, ImageTk, ImageDraw
from .constants import THUMBNAIL_SIZE, CORNER_RADIUS


def create_rounded_image(pil_image, size, radius=CORNER_RADIUS):
    """Apply a rounded mask to a PIL image."""
    pil_image = pil_image.resize((size, size), Image.LANCZOS)
    mask = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle([(0, 0), (size - 1, size - 1)], radius, fill=255)
    output = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    output.paste(pil_image.convert("RGBA"), mask=mask)
    return output


def numpy_to_photo(image_np, size=THUMBNAIL_SIZE):
    """Convert a numpy array (0-1, 64x64) to a PhotoImage with rounded corners."""
    image_uint8 = (np.clip(image_np, 0, 1) * 255).astype(np.uint8)
    pil_image = Image.fromarray(image_uint8, mode="L").convert("RGB")
    rounded = create_rounded_image(pil_image, size)
    return ImageTk.PhotoImage(rounded)
