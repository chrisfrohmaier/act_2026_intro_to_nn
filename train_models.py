"""Train the two demo networks on MNIST with PyTorch and save their weights for the app.

    python train_models.py --data path/to/mnist_folder

The folder must contain the four MNIST IDX files (.gz or not), e.g. the MNIST/raw
folder that torchvision.datasets.MNIST(download=True) creates.
You only need this to retrain; the app ships with the trained weights in models/.
Takes a few minutes on a laptop CPU.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from nncore.data import load_mnist, shift_images

OUT = Path(__file__).resolve().parent / "models"


class MLP(nn.Module):
    """784 → 16 → 16 → 10, ReLU: the classic small network for MNIST."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 16), nn.ReLU(),
            nn.Linear(16, 16), nn.ReLU(),
            nn.Linear(16, 10),
        )

    def forward(self, x):
        return self.net(x)


class CNN(nn.Module):
    """Two convolution + pooling stages, then one dense layer: about the same number of weights as the MLP."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool2d(2),   # 28×28 → 14×14
            nn.Conv2d(8, 16, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool2d(2),  # 14×14 → 7×7
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(16 * 7 * 7, 10))

    def forward(self, x):
        return self.classifier(self.features(x))


def train(model, loader, epochs, lr=1e-3):
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    history = []
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)
        history.append(total / len(loader.dataset))
        print(f"  epoch {epoch + 1}: loss {history[-1]:.4f}")
    return history


@torch.no_grad()
def accuracy(model, x, y):
    model.eval()
    pred = model(torch.as_tensor(x, dtype=torch.float32)).argmax(1).numpy()
    return float((pred == y).mean())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--epochs-mlp", type=int, default=15)
    p.add_argument("--epochs-cnn", type=int, default=8)
    args = p.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    x_tr, y_tr, x_te, y_te = load_mnist(args.data)
    x_tr = (x_tr / 255.0).astype(np.float32)[:, None]   # (n, 1, 28, 28), values 0–1
    x_te = (x_te / 255.0).astype(np.float32)[:, None]
    loader = DataLoader(TensorDataset(torch.tensor(x_tr), torch.tensor(y_tr, dtype=torch.long)),
                        batch_size=64, shuffle=True)

    print("Training MLP")
    mlp = MLP()
    mlp_hist = train(mlp, loader, args.epochs_mlp)
    print("Training CNN")
    cnn = CNN()
    cnn_hist = train(cnn, loader, args.epochs_cnn)

    OUT.mkdir(exist_ok=True)
    sd = {k: v.numpy() for k, v in mlp.state_dict().items()}
    np.savez(OUT / "mlp.npz", W1=sd["net.1.weight"], b1=sd["net.1.bias"], W2=sd["net.3.weight"],
             b2=sd["net.3.bias"], W3=sd["net.5.weight"], b3=sd["net.5.bias"])
    sd = {k: v.numpy() for k, v in cnn.state_dict().items()}
    np.savez(OUT / "cnn.npz", K1=sd["features.0.weight"], c1=sd["features.0.bias"],
             K2=sd["features.3.weight"], c2=sd["features.3.bias"],
             Wf=sd["classifier.1.weight"], bf=sd["classifier.1.bias"])

    # How accuracy falls when every test digit is moved off-centre (the shift test)
    shifts = list(range(0, 9))
    curve = {"shift_px": shifts, "mlp": [], "cnn": []}
    for s in shifts:
        accs_m, accs_c = [], []
        for dx, dy in ((s, 0), (-s, 0), (0, s), (0, -s)):   # right, left, down, up
            xs = shift_images(x_te, dx, dy)
            accs_m.append(accuracy(mlp, xs, y_te))
            accs_c.append(accuracy(cnn, xs, y_te))
        curve["mlp"].append(float(np.mean(accs_m)))
        curve["cnn"].append(float(np.mean(accs_c)))
        print(f"shift {s}px: MLP {curve['mlp'][-1]:.3f}  CNN {curve['cnn'][-1]:.3f}")

    stats = {
        "mlp": {"test_accuracy": curve["mlp"][0], "params": sum(p.numel() for p in mlp.parameters()),
                "loss_history": mlp_hist, "epochs": args.epochs_mlp},
        "cnn": {"test_accuracy": curve["cnn"][0], "params": sum(p.numel() for p in cnn.parameters()),
                "loss_history": cnn_hist, "epochs": args.epochs_cnn},
        "shift_curve": curve,
    }
    (OUT / "stats.json").write_text(json.dumps(stats, indent=2))

    # A few real test digits for the app's preset buttons (the first of each class)
    idx = [int(np.where(y_te == d)[0][0]) for d in range(10)]
    np.savez(OUT / "examples.npz", images=(x_te[idx, 0] * 255).astype(np.uint8), labels=y_te[idx])
    print(json.dumps({k: v["test_accuracy"] for k, v in stats.items() if k != "shift_curve"}, indent=2))


if __name__ == "__main__":
    main()
