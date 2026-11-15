import numpy as np
import streamlit as st

from nncore import plots
from nncore.models import CNN, conv2d, maxpool2, relu
from nncore.ui import (current_digit, describe_source, digit_picker, load_models, load_stats, page_header,
                       show_code, status)

page_header(
    "Demo 4",
    "Inside the CNN",
    "A convolution slides a small grid of weights (a filter) across the image. At every position it takes the "
    "weighted sum of the pixels underneath, so one filter produces a whole new image: a feature map showing "
    "where its pattern appears. ReLU keeps the positive responses; max pooling keeps the strongest response in "
    "each 2 × 2 block, halving the size.",
)
mlp, cnn = load_models()
stats = load_stats()

with st.expander("Change the digit", expanded=False):
    digit_picker("cnn", canvas_size=200)
img = current_digit()

# ---------------------------------------------------------------- 1. a hand-made filter
st.header("1 · One filter, by hand")
FILTERS = {
    "Vertical edge": [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]],
    "Horizontal edge": [[-1, -1, -1], [0, 0, 0], [1, 1, 1]],
    "Diagonal ↘": [[1, 0, -1], [0, 1, 0], [-1, 0, 1]],
    "Blur": [[1 / 9] * 3] * 3,
}
name = st.radio("Filter", list(FILTERS), horizontal=True)
K = np.array(FILTERS[name], dtype=np.float32)
fmap = conv2d(img[None], K[None, None], np.zeros(1))[0]
rel = relu(fmap)
pooled = maxpool2(rel[None])[0]

c0, c1, c2, c3, c4 = st.columns(5)
with c0:
    st.pyplot(plots.digit(img, "input 28 × 28", size=2.3), clear_figure=True)
with c1:
    st.markdown(f"**{name} filter (3 × 3)**")
    st.dataframe(np.round(K, 2), hide_index=True)
with c2:
    st.pyplot(plots.maps(fmap[None], ncols=1, cmap=plots.DIV, symmetric=True, titles=["convolved 28 × 28"],
                         size_per=2.3), clear_figure=True)
with c3:
    st.pyplot(plots.maps(rel[None], ncols=1, titles=["after ReLU"], size_per=2.3), clear_figure=True)
with c4:
    st.pyplot(plots.maps(pooled[None], ncols=1, titles=["max-pooled 14 × 14"], size_per=2.3), clear_figure=True)
st.caption("The same 9 weights are used at every position: that is what makes a CNN cheap, and why a stroke is "
           "detected wherever it is.")

# ---------------------------------------------------------------- 2. the trained network
st.header("2 · What the trained CNN learned")
out = cnn.forward(img)
best = int(np.argmax(out["probs"]))
status(f"For {describe_source()}, the CNN says <b>{best}</b>, {out['probs'][best]:.0%} sure.")

st.markdown("**The 8 learned first-layer filters (5 × 5).** Nobody designed these: training found them. "
            "Several look like edge and stroke detectors.")
st.pyplot(plots.maps(cnn.K1[:, 0], ncols=8, cmap=plots.DIV, symmetric=True,
                     titles=[f"filter {i + 1}" for i in range(8)], size_per=1.2), clear_figure=True)
st.markdown("**What each filter sees in this digit:** 8 feature maps, 28 × 28, after ReLU.")
st.pyplot(plots.maps(out["conv1"], ncols=8, titles=[f"map {i + 1}" for i in range(8)], size_per=1.2),
          clear_figure=True)

with st.expander("Deeper: layer 2 and pooling", expanded=False):
    st.markdown("After pooling to 14 × 14, layer 2 combines all 8 maps with 16 new filters (each 8 × 5 × 5). "
                "Its maps respond to combinations of strokes: corners, loops, endings.")
    st.pyplot(plots.maps(out["conv2"], ncols=8, titles=[f"map {i + 1}" for i in range(16)], size_per=1.1),
              clear_figure=True)
    st.markdown("Pooled again to 16 maps of 7 × 7 = 784 numbers, which a single dense layer turns into 10 scores.")
    st.pyplot(plots.prob_bars(out["probs"], plots.CNN_COLOUR, "CNN output", size=(5, 2.4)), clear_figure=True)

# ---------------------------------------------------------------- 3. the comparison
st.header("3 · Same budget, better network")
st.table({
    "": ["Fully connected (MLP)", "Convolutional (CNN)"],
    "Layers": ["784 → 16 → 16 → 10", "conv 8 → pool → conv 16 → pool → 10"],
    "Weights and biases": [f"{stats['mlp']['params']:,}", f"{stats['cnn']['params']:,}"],
    "Test accuracy": [f"{stats['mlp']['test_accuracy']:.1%}", f"{stats['cnn']['test_accuracy']:.1%}"],
    "Accuracy, digits moved 3 px": [f"{stats['shift_curve']['mlp'][3]:.0%}",
                                    f"{stats['shift_curve']['cnn'][3]:.0%}"],
})
st.caption(f"Errors on the test set: MLP {round((1 - stats['mlp']['test_accuracy']) * 10000)} of 10,000, "
           f"CNN {round((1 - stats['cnn']['test_accuracy']) * 10000)} of 10,000.")

show_code(conv2d, maxpool2, CNN.forward,
          note="`conv2d` gathers every 5 × 5 patch, then one matrix multiply applies all the filters at once.")
