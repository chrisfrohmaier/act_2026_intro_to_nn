"""Turn a drawing from the canvas into an MNIST-style 28×28 image.

MNIST digits were prepared like this: crop to the ink, scale so the longest side
is 20 pixels, then paste into a 28×28 frame so the centre of mass sits in the middle.
Doing the same to your drawing makes it look like what the networks were trained on.
"""
import numpy as np
from PIL import Image


def canvas_to_mnist(rgba):
    """rgba: (H, W, 4) uint8 array from the drawing canvas (white ink on black).

    Returns a (28, 28) float array in 0–1, or None if nothing has been drawn.
    """
    if rgba is None:
        return None
    ink = rgba[..., :3].astype(np.float32).mean(axis=2) * (rgba[..., 3] / 255.0)
    if ink.max() < 30:
        return None
    rows = np.where(ink.max(axis=1) > 30)[0]
    cols = np.where(ink.max(axis=0) > 30)[0]
    crop = ink[rows[0]: rows[-1] + 1, cols[0]: cols[-1] + 1]

    h, w = crop.shape
    scale = 20.0 / max(h, w)
    new_h, new_w = max(1, round(h * scale)), max(1, round(w * scale))
    small = np.asarray(Image.fromarray(crop.astype(np.uint8)).resize((new_w, new_h), Image.LANCZOS),
                       dtype=np.float32)
    small = np.clip(small, 0, 255)

    out = np.zeros((28, 28), dtype=np.float32)
    top, left = (28 - new_h) // 2, (28 - new_w) // 2
    out[top: top + new_h, left: left + new_w] = small

    # Move the centre of mass to the middle of the frame
    total = out.sum()
    yy, xx = np.indices(out.shape)
    cy, cx = (yy * out).sum() / total, (xx * out).sum() / total
    out = shift(out, int(round(13.5 - cx)), int(round(13.5 - cy)))
    return out / max(out.max(), 1e-6)


def shift(image, dx, dy):
    """Move a 2-D image dx pixels right and dy pixels down; pixels pushed off the edge are lost."""
    out = np.zeros_like(image)
    h, w = image.shape
    if abs(dx) >= w or abs(dy) >= h:
        return out
    out[max(0, dy): h + min(0, dy), max(0, dx): w + min(0, dx)] = \
        image[max(0, -dy): h - max(0, dy), max(0, -dx): w - max(0, dx)]
    return out
