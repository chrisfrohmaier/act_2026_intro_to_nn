import numpy as np
import streamlit as st

from nncore import plots
from nncore.ui import current_digit, describe_source, digit_picker, load_models, page_header, show_code, status

page_header(
    "Demo 2",
    "What the hidden neurons look for",
    "Each neuron in the first hidden layer has 784 weights, one per pixel. Draw them back as a 28 × 28 picture "
    "and you can see the pattern it responds to: <span style='color:#2a6fc9'><b>blue</b></span> pixels push it "
    "to fire, <span style='color:#b04618'><b>orange</b></span> pixels hold it back.",
)
mlp, _ = load_models()

with st.expander("Change the digit", expanded=False):
    digit_picker("neurons", canvas_size=200)
img = current_digit()
acts = mlp.forward(img)
x = acts["input"]
z1 = mlp.W1 @ x + mlp.b1
fired = np.maximum(0, z1)
k = int(np.argmax(fired))

status(f"For {describe_source()}, neuron <b>{k + 1}</b> fires hardest ({fired[k]:.1f}). "
       f"{(fired > 0).sum()} of the 16 neurons fire at all.")

st.subheader("The 16 first-layer neurons, as pictures")
st.pyplot(plots.weight_images(mlp.W1, contributions=fired), clear_figure=True)
st.caption("“fires” is ReLU(weights · pixels + bias) for the current digit.")

st.subheader(f"Inside one neuron: neuron {k + 1}")
choice = st.select_slider("Neuron", options=list(range(1, 17)), value=k + 1)
j = choice - 1
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.pyplot(plots.digit(img, "pixels x"), clear_figure=True)
with c2:
    st.pyplot(plots.maps(mlp.W1[j].reshape(1, 28, 28), ncols=1, cmap=plots.DIV, symmetric=True,
                         titles=["weights w"], size_per=2.4), clear_figure=True)
with c3:
    st.pyplot(plots.maps((mlp.W1[j] * x).reshape(1, 28, 28), ncols=1, cmap=plots.DIV, symmetric=True,
                         titles=["w × x, pixel by pixel"], size_per=2.4), clear_figure=True)
with c4:
    st.markdown(f"""
**Sum of w × x:** {mlp.W1[j] @ x:+.2f}

**Plus bias b:** {mlp.b1[j]:+.2f}

**= weighted sum z:** {z1[j]:+.2f}

**ReLU(z):** {fired[j]:.2f}
""")
st.info("Notice the pictures are not neat strokes or loops. The network found whatever weights lowered the "
        "loss, not features a human would draw. Also, every neuron is tied to fixed pixel positions: "
        "move the digit and it is looking in the wrong place. That is Demo 3.")

show_code(type(mlp).forward, note="Hidden layer 1 is `relu(self.W1 @ x + self.b1)`: row j of W1 is the picture above.")
