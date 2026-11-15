"""Load MNIST from the original IDX files (the same files torchvision downloads).

Plain NumPy, no Streamlit and no PyTorch, so it can be reused anywhere.
"""
import gzip
from pathlib import Path

import numpy as np


def _read_idx(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as f:
        data = f.read()
    ndim = data[3]
    shape = tuple(int.from_bytes(data[4 + 4 * i: 8 + 4 * i], "big") for i in range(ndim))
    return np.frombuffer(data, dtype=np.uint8, offset=4 + 4 * ndim).reshape(shape)


def load_mnist(folder):
    """Return (x_train, y_train, x_test, y_test); images are uint8 arrays of shape (n, 28, 28)."""
    folder = Path(folder)

    def find(stem):
        for name in (stem + ".gz", stem):
            if (folder / name).exists():
                return folder / name
        raise FileNotFoundError(f"{stem}(.gz) not found in {folder}")

    return (
        _read_idx(find("train-images-idx3-ubyte")),
        _read_idx(find("train-labels-idx1-ubyte")),
        _read_idx(find("t10k-images-idx3-ubyte")),
        _read_idx(find("t10k-labels-idx1-ubyte")),
    )


def shift_images(images, dx, dy):
    """Shift a batch of (n, 28, 28) images by dx pixels right and dy pixels down, filling with zeros."""
    out = np.zeros_like(images)
    h, w = images.shape[-2:]
    src_y = slice(max(0, -dy), min(h, h - dy))
    dst_y = slice(max(0, dy), min(h, h + dy))
    src_x = slice(max(0, -dx), min(w, w - dx))
    dst_x = slice(max(0, dx), min(w, w + dx))
    out[..., dst_y, dst_x] = images[..., src_y, src_x]
    return out
