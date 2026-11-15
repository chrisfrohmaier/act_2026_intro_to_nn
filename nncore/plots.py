"""Matplotlib figures for the app, in the lecture slides' colours."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

PAPER = "#f5f3ed"
INK = "#1c2130"
MUTED = "#6a7080"
GRID = "#d8d4c8"
TEAL = "#1f6b66"
TEAL_LIGHT = "#cfe3e0"
MLP_COLOUR = "#b04618"   # orange: the fully connected network
CNN_COLOUR = "#2a6fc9"   # blue: the convolutional network
NEG = "#b04618"
POS = "#2a6fc9"

SEQ = LinearSegmentedColormap.from_list("seq", ["#fffdf8", TEAL_LIGHT, TEAL, "#0d3532"])
DIV = LinearSegmentedColormap.from_list("div", [NEG, "#f0c9b4", "#ecebe6", "#bcd1ef", POS])
INK_CMAP = LinearSegmentedColormap.from_list("ink", ["#fffdf8", INK])

def _fresh(fn):
    """Close earlier figures (already sent to the browser) before drawing a new one."""
    def wrapper(*args, **kwargs):
        plt.close("all")
        return fn(*args, **kwargs)
    wrapper.__name__, wrapper.__doc__ = fn.__name__, fn.__doc__
    return wrapper


plt.rcParams.update({
    "font.size": 11, "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED,
    "ytick.color": MUTED, "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": PAPER, "axes.facecolor": PAPER, "savefig.facecolor": PAPER,
})


@_fresh
def digit(img, title=None, size=2.6):
    fig, ax = plt.subplots(figsize=(size, size))
    ax.imshow(img, cmap=INK_CMAP, vmin=0, vmax=1)
    ax.set_xticks([]), ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True), s.set_color(GRID)
    if title:
        ax.set_title(title, fontsize=10, color=MUTED)
    fig.tight_layout()
    return fig


@_fresh
def prob_bars(probs, colour=TEAL, title=None, size=(4.6, 2.6)):
    fig, ax = plt.subplots(figsize=size)
    best = int(np.argmax(probs))
    colours = [colour if i == best else "#c9c6bb" for i in range(10)]
    ax.bar(range(10), probs, color=colours, width=0.7)
    ax.set_xticks(range(10))
    ax.set_ylim(0, 1.08)
    ax.set_yticks([0, 0.5, 1])
    ax.set_yticklabels(["0%", "50%", "100%"])
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.text(best, probs[best] + 0.03, f"{probs[best]:.0%}", ha="center", fontsize=10, color=INK)
    if title:
        ax.set_title(title, fontsize=11, loc="left", color=INK)
    ax.set_xlabel("digit")
    fig.tight_layout()
    return fig


@_fresh
def network(acts, mlp, img, top_edges=60, size=(9.5, 5.2)):
    """The 784 → 16 → 16 → 10 network with this digit's activations lit up."""
    fig, ax = plt.subplots(figsize=size)
    ax.set_xlim(-0.6, 3.35), ax.set_ylim(-0.6, 16.2)
    ax.axis("off")

    def ys(n):
        return np.linspace(15.2, 0.2, n) if n == 16 else np.linspace(13.2, 2.2, n)

    xs = [0.0, 1.15, 2.1, 3.0]
    layers = [acts["hidden1"], acts["hidden2"], acts["probs"]]
    weights = [None, mlp.W2, mlp.W3]

    # Input: draw the image itself, with a few lines fanning into hidden layer 1
    ax.imshow(img, cmap=INK_CMAP, vmin=0, vmax=1, extent=(-0.55, 0.0, 5.8, 10.63), zorder=3,
              aspect="auto")
    ax.text(-0.275, 5.4, "784 inputs\n(28 × 28 pixels)", ha="center", va="top", fontsize=9, color=MUTED)
    h1 = acts["hidden1"]
    for j, y in enumerate(ys(16)):
        a = h1[j] / (h1.max() + 1e-9)
        ax.plot([0.02, xs[1]], [8.2, y], color=TEAL, alpha=0.08 + 0.5 * a, lw=0.6 + 1.6 * a, zorder=1)

    # Connections between later layers: show the strongest contributions w × a
    for li in (1, 2):
        src, W = layers[li - 1], weights[li]
        contrib = W * src[None, :]
        flat = np.argsort(np.abs(contrib).ravel())[::-1][:top_edges]
        cmax = np.abs(contrib).max() + 1e-9
        ys_src, ys_dst = ys(len(src)), ys(W.shape[0])
        for k in flat:
            i, j = divmod(k, W.shape[1])
            c = contrib[i, j]
            ax.plot([xs[li], xs[li + 1]], [ys_src[j], ys_dst[i]], color=POS if c > 0 else NEG,
                    alpha=0.15 + 0.6 * abs(c) / cmax, lw=0.5 + 2.2 * abs(c) / cmax, zorder=1)

    # Neurons
    for li, vals in enumerate(layers):
        norm = vals / (vals.max() + 1e-9) if li < 2 else vals
        for k, (y, v) in enumerate(zip(ys(len(vals)), norm)):
            ax.scatter(xs[li + 1], y, s=230 if li < 2 else 330, color=SEQ(0.05 + 0.95 * v),
                       edgecolor=TEAL, lw=1.2, zorder=4)
            if li == 2:
                ax.text(xs[3] + 0.17, y, f"{k}", va="center", fontsize=12,
                        fontweight="bold" if k == np.argmax(vals) else "normal",
                        color=INK if k == np.argmax(vals) else MUTED)
    labels = ["hidden layer 1\n16 neurons", "hidden layer 2\n16 neurons", "output\n10 digits"]
    for x, lab in zip(xs[1:], labels):
        ax.text(x, -0.55, lab, ha="center", va="top", fontsize=9, color=MUTED)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.12)
    return fig


@_fresh
def weight_images(W1, contributions=None, size=(9.5, 2.9)):
    """Each first-layer neuron's 784 weights, drawn back as a 28×28 picture."""
    fig, axes = plt.subplots(2, 8, figsize=size)
    lim = np.abs(W1).max()
    for k, ax in enumerate(axes.ravel()):
        ax.imshow(W1[k].reshape(28, 28), cmap=DIV, vmin=-lim, vmax=lim)
        ax.set_xticks([]), ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(True), s.set_color(GRID)
        label = f"neuron {k + 1}"
        if contributions is not None:
            label += f"\nfires {contributions[k]:.1f}"
        ax.set_title(label, fontsize=8, color=MUTED)
    fig.tight_layout(pad=0.4)
    return fig


@_fresh
def maps(stack, ncols=8, cmap=SEQ, titles=None, size_per=1.15, symmetric=False):
    n = stack.shape[0]
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * size_per, nrows * size_per * 1.3))
    axes = np.atleast_1d(axes).ravel()
    lim = np.abs(stack).max() + 1e-9
    for k, ax in enumerate(axes):
        ax.axis("off")
        if k < n:
            if symmetric:
                ax.imshow(stack[k], cmap=cmap, vmin=-lim, vmax=lim)
            else:
                ax.imshow(stack[k], cmap=cmap, vmin=0, vmax=max(stack[k].max(), 1e-9))
            if titles:
                ax.set_title(titles[k], fontsize=8, color=MUTED)
    fig.tight_layout(pad=0.3)
    return fig


@_fresh
def shift_curve(curve, current=None, size=(5.6, 3.1)):
    fig, ax = plt.subplots(figsize=size)
    s = curve["shift_px"]
    ax.plot(s, curve["mlp"], color=MLP_COLOUR, lw=2, marker="o", ms=5, label="Fully connected (MLP)")
    ax.plot(s, curve["cnn"], color=CNN_COLOUR, lw=2, marker="s", ms=5, label="Convolutional (CNN)")
    ax.text(s[-1] + 0.15, curve["mlp"][-1], "MLP", color=INK, va="center", fontsize=10)
    ax.text(s[-1] + 0.15, curve["cnn"][-1], "CNN", color=INK, va="center", fontsize=10)
    if current is not None:
        ax.axvline(current, color=MUTED, lw=1, ls="--")
    ax.set_ylim(0, 1.02)
    ax.set_xlim(-0.3, s[-1] + 1.2)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("shift in pixels (test digits moved off-centre)")
    ax.set_ylabel("test accuracy")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    fig.tight_layout()
    return fig
