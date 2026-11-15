import numpy as np
import streamlit as st

from nncore import plots
from nncore.models import MLP, relu, softmax
from nncore.ui import describe_source, digit_picker, load_models, page_header, show_code, status

page_header(
    "Demo 1",
    "Draw a digit",
    "The fully connected network sees your digit as 784 numbers. Each layer takes a weighted sum of the "
    "layer before, adds a bias and applies ReLU. The last layer gives one score per digit, and softmax turns "
    "the scores into probabilities.",
)
mlp, _ = load_models()

left, right = st.columns([1, 1.25], gap="large")
with left:
    img = digit_picker("draw")
acts = mlp.forward(img)
probs = acts["probs"]
best = int(np.argmax(probs))
runner_up = int(np.argsort(probs)[-2])

with right:
    status(f"The network says <b>{best}</b>, {probs[best]:.0%} sure. "
           f"Second choice: {runner_up} ({probs[runner_up]:.0%}). Input: {describe_source()}.")
    a, b = st.columns([1, 1.6])
    with a:
        st.pyplot(plots.digit(img, "what the network sees: 28 × 28"), clear_figure=True)
    with b:
        st.pyplot(plots.prob_bars(probs, title="softmax output"), clear_figure=True)
    st.markdown(
        f"Hidden layer 1: **{(acts['hidden1'] > 0).sum()} of 16** neurons active · "
        f"hidden layer 2: **{(acts['hidden2'] > 0).sum()} of 16** active. "
        "ReLU sets every negative weighted sum to exactly zero, so the rest stay dark."
    )

st.subheader("The whole network, for this digit")
st.caption("Darker neurons fired more strongly. Lines show the 60 strongest contributions (weight × activation) "
           "into the next layer: blue pushes the next neuron up, orange pushes it down.")
st.pyplot(plots.network(acts, mlp, img), clear_figure=True)

with st.expander("The numbers"):
    c1, c2, c3 = st.columns(3)
    c1.markdown("**Hidden layer 1**")
    c1.dataframe({"neuron": range(1, 17), "activation": np.round(acts["hidden1"], 2)}, hide_index=True)
    c2.markdown("**Hidden layer 2**")
    c2.dataframe({"neuron": range(1, 17), "activation": np.round(acts["hidden2"], 2)}, hide_index=True)
    c3.markdown("**Output**")
    c3.dataframe({"digit": range(10), "score": np.round(acts["scores"], 2),
                  "probability": np.round(probs, 3)}, hide_index=True)
    st.markdown(f"This network has **{mlp.n_params:,}** weights and biases: "
                "784×16 + 16 + 16×16 + 16 + 16×10 + 10.")

show_code(MLP.forward, relu, softmax,
          note="The forward pass is three matrix multiplications. `@` is matrix multiplication in NumPy.")
