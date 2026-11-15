import streamlit as st

from nncore.ui import load_stats, page_header

page_header(
    "ACT · Neural networks",
    "Neural Network Playground",
    "Two small networks trained on MNIST, the handwritten digits you will use in this week's notebooks. "
    "Draw a digit and watch them read it.",
)
stats = load_stats()
c1, c2 = st.columns(2)
with c1:
    st.subheader("The fully connected network (MLP)")
    st.markdown(
        "784 pixels → 16 neurons → 16 neurons → 10 outputs, with ReLU between layers. "
        "The same kind of network you build in the notebook."
    )
    st.metric("Test accuracy", f"{stats['mlp']['test_accuracy']:.1%}")
    st.metric("Weights and biases", f"{stats['mlp']['params']:,}")
with c2:
    st.subheader("The convolutional network (CNN)")
    st.markdown(
        "Two rounds of convolution (5×5 filters), ReLU and 2×2 max pooling, then one dense layer to 10 outputs."
    )
    st.metric("Test accuracy", f"{stats['cnn']['test_accuracy']:.1%}")
    st.metric("Weights and biases", f"{stats['cnn']['params']:,}")

st.markdown(
    """
### The demos
1. **Draw a digit**: the MLP's activations light up layer by layer, and the output becomes ten probabilities.
2. **What the neurons look for**: each hidden neuron's 784 weights drawn back as a picture.
3. **The shift test**: move the digit a few pixels and see which network copes.
4. **Inside the CNN**: the filters it learned, and what each one sees in your digit.

Your digit is shared between pages, so draw once and then visit each demo.
Every page has a **Show the code** box with the real NumPy behind it.
Both networks were trained in PyTorch (`train_models.py`) and run here in plain NumPy (`nncore/models.py`).
"""
)
