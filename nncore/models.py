"""The two trained networks, run forward in plain NumPy.

The weights were trained in PyTorch (see train_models.py) and saved as .npz files.
Running them here in NumPy shows there is no magic: a trained network is just
matrix multiplications, additions and a simple non-linearity.
"""
from pathlib import Path

import numpy as np

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"


def relu(z):
    return np.maximum(0.0, z)


def softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


# ---------------------------------------------------------------------------
# Fully connected network (multilayer perceptron): 784 → 16 → 16 → 10
# ---------------------------------------------------------------------------
class MLP:
    def __init__(self, path=MODEL_DIR / "mlp.npz"):
        w = np.load(path)
        self.W1, self.b1 = w["W1"], w["b1"]   # (16, 784), (16,)
        self.W2, self.b2 = w["W2"], w["b2"]   # (16, 16),  (16,)
        self.W3, self.b3 = w["W3"], w["b3"]   # (10, 16),  (10,)

    @property
    def n_params(self):
        return sum(a.size for a in (self.W1, self.b1, self.W2, self.b2, self.W3, self.b3))

    def forward(self, image):
        """image: (28, 28) array with values 0–1. Returns every layer's activations."""
        x = image.reshape(784)                 # flatten: the grid becomes one long vector
        a1 = relu(self.W1 @ x + self.b1)       # hidden layer 1
        a2 = relu(self.W2 @ a1 + self.b2)      # hidden layer 2
        scores = self.W3 @ a2 + self.b3        # one score per digit
        probs = softmax(scores)                # scores → probabilities that sum to 1
        return {"input": x, "hidden1": a1, "hidden2": a2, "scores": scores, "probs": probs}


# ---------------------------------------------------------------------------
# Convolutional network:
#   conv 5×5 (8 filters) → ReLU → max-pool 2×2 →
#   conv 5×5 (16 filters) → ReLU → max-pool 2×2 → fully connected → 10
# ---------------------------------------------------------------------------
def conv2d(x, W, b):
    """x: (C_in, H, W) image stack; W: (C_out, C_in, k, k). 'Same' padding, stride 1."""
    c_out, c_in, k, _ = W.shape
    pad = k // 2
    xp = np.pad(x, ((0, 0), (pad, pad), (pad, pad)))
    H, Wd = x.shape[1:]
    # Collect every k×k patch (im2col), then one matrix multiply does all the filters at once
    patches = np.lib.stride_tricks.sliding_window_view(xp, (k, k), axis=(1, 2))  # (C_in, H, W, k, k)
    patches = patches.transpose(1, 2, 0, 3, 4).reshape(H * Wd, c_in * k * k)
    out = patches @ W.reshape(c_out, -1).T + b
    return out.T.reshape(c_out, H, Wd)


def maxpool2(x):
    """2×2 max pooling: keep the strongest response in each 2×2 block."""
    c, h, w = x.shape
    return x[:, : h // 2 * 2, : w // 2 * 2].reshape(c, h // 2, 2, w // 2, 2).max(axis=(2, 4))


class CNN:
    def __init__(self, path=MODEL_DIR / "cnn.npz"):
        w = np.load(path)
        self.K1, self.c1 = w["K1"], w["c1"]   # (8, 1, 5, 5)
        self.K2, self.c2 = w["K2"], w["c2"]   # (16, 8, 5, 5)
        self.Wf, self.bf = w["Wf"], w["bf"]   # (10, 16*7*7)

    @property
    def n_params(self):
        return sum(a.size for a in (self.K1, self.c1, self.K2, self.c2, self.Wf, self.bf))

    def forward(self, image):
        x = image[None, :, :]                          # (1, 28, 28): one input channel
        f1 = relu(conv2d(x, self.K1, self.c1))         # (8, 28, 28): 8 feature maps
        p1 = maxpool2(f1)                              # (8, 14, 14)
        f2 = relu(conv2d(p1, self.K2, self.c2))        # (16, 14, 14)
        p2 = maxpool2(f2)                              # (16, 7, 7)
        scores = self.Wf @ p2.reshape(-1) + self.bf    # flatten, then a normal dense layer
        probs = softmax(scores)
        return {"conv1": f1, "pool1": p1, "conv2": f2, "pool2": p2, "scores": scores, "probs": probs}
