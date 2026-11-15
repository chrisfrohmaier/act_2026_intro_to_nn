# Neural Network Playground

Live demos for the ACT lecture on neural networks. Students draw a digit and watch two small
networks read it. Both networks were trained on MNIST, the dataset used in the notebooks.

| Page | What it shows |
|---|---|
| 1 · Draw a digit | A 784 → 16 → 16 → 10 network with this digit's activations lit up, and the softmax output |
| 2 · What the neurons look for | Each first-layer neuron's 784 weights drawn as a 28 × 28 picture, and one neuron's weighted sum worked through |
| 3 · The shift test | Move the digit a few pixels: the MLP falls apart, the CNN copes (▶ Play slides it across) |
| 4 · Inside the CNN | A hand-made filter (convolution → ReLU → max pool), the 8 filters the CNN learned, and the feature maps for your digit |

The digit is shared between pages: draw it once on page 1, then visit the others.

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

Needs Python 3.10+ and Streamlit 1.53 or newer (for the drawing canvas). No PyTorch is needed to run the app:
the trained weights are in `models/` and the networks run forward in plain NumPy (`nncore/models.py`).

### Streamlit Community Cloud

Push this folder to a GitHub repo, then on share.streamlit.io choose **New app** and set the main file to `app.py`.
Apps go to sleep after 12 hours without visitors, so open it about 10 minutes before the lecture.

## The two networks

| | MLP | CNN |
|---|---|---|
| Layers | 784 → 16 → 16 → 10, ReLU | conv 5×5 ×8 → ReLU → pool 2×2 → conv 5×5 ×16 → ReLU → pool 2×2 → dense → 10 |
| Weights and biases | 13,002 | 11,274 |
| Test accuracy | 95.2% | 98.7% |
| Accuracy with digits moved 3 px | 40% | 86% |

Trained with Adam (learning rate 0.001, batch size 64): 15 epochs for the MLP, 8 for the CNN. Neither saw shifted
digits in training. The shift curve averages moves up, down, left and right over the full 10,000-image test set.

## Retrain

```bash
pip install torch
python train_models.py --data path/to/MNIST/raw
```

`--data` is any folder holding the four MNIST IDX files, such as the `MNIST/raw` folder that
`torchvision.datasets.MNIST(download=True)` creates in the notebook. Takes a few minutes on a laptop CPU and
rewrites everything in `models/` (weights, `stats.json`, and the example digits).

## Layout

```
app.py                 navigation
pages/                 one file per demo
nncore/models.py       the forward passes in NumPy (shown in "Show the code")
nncore/preprocess.py   canvas drawing → MNIST-style 28 × 28 (crop, scale to 20 px, centre of mass)
nncore/plots.py        matplotlib figures in the lecture colours
nncore/ui.py           shared digit picker and page furniture
train_models.py        PyTorch training script
models/                trained weights and stats
```
