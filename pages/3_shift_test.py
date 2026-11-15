import time

import numpy as np
import streamlit as st

from nncore import plots
from nncore.preprocess import shift
from nncore.ui import (current_digit, describe_source, digit_picker, load_models, load_stats, page_header,
                       show_code, status)

page_header(
    "Demo 3",
    "The shift test",
    "Move the same digit a few pixels. To a person it is the same 7. To the fully connected network, almost "
    "every one of its 784 inputs has changed. The convolutional network slides the same small filters over "
    "the whole image, so a shifted stroke still triggers the same filter, just somewhere else.",
)
mlp, cnn = load_models()
stats = load_stats()

with st.expander("Change the digit", expanded=False):
    digit_picker("shift", canvas_size=200)
img = current_digit()

PRESETS = {"Centred": (0, 0), "2 px right": (2, 0), "4 px right": (4, 0), "3 px down-left": (-3, 3),
           "6 px up": (0, -6)}


def set_shift(dx, dy):
    st.session_state["dx"], st.session_state["dy"] = dx, dy


cols = st.columns(len(PRESETS) + 1)
for c, (name, (dx, dy)) in zip(cols, PRESETS.items()):
    c.button(name, on_click=set_shift, args=(dx, dy), use_container_width=True)
play = cols[-1].button("▶ Play: slide right", type="primary", use_container_width=True)

st.session_state.setdefault("dx", 0)
st.session_state.setdefault("dy", 0)
s1, s2 = st.columns(2)
dx = s1.slider("Move right (pixels)", -8, 8, key="dx")
dy = s2.slider("Move down (pixels)", -8, 8, key="dy")

status_box = st.empty()
row = st.columns([1, 1.4, 1.4])
digit_box, mlp_box, cnn_box = (c.empty() for c in row)
curve_box = st.empty()


def render(dx, dy):
    moved = shift(img, dx, dy)
    pm, pc = mlp.forward(moved)["probs"], cnn.forward(moved)["probs"]
    bm, bc = int(np.argmax(pm)), int(np.argmax(pc))
    where = "centred" if dx == dy == 0 else f"moved {dx:+d} px across, {dy:+d} px down"
    status_box.markdown(
        f'<div class="status">{describe_source()}, {where}: '
        f'<b style="color:{plots.MLP_COLOUR}">MLP says {bm}</b> ({pm[bm]:.0%}) · '
        f'<b style="color:{plots.CNN_COLOUR}">CNN says {bc}</b> ({pc[bc]:.0%})</div>',
        unsafe_allow_html=True)
    digit_box.pyplot(plots.digit(moved, f"shifted ({dx:+d}, {dy:+d})", size=2.8), clear_figure=True)
    mlp_box.pyplot(plots.prob_bars(pm, plots.MLP_COLOUR, "Fully connected (MLP)"), clear_figure=True)
    cnn_box.pyplot(plots.prob_bars(pc, plots.CNN_COLOUR, "Convolutional (CNN)"), clear_figure=True)
    curve_box.pyplot(plots.shift_curve(stats["shift_curve"], current=max(abs(dx), abs(dy))),
                     clear_figure=True)


if play:
    for step in range(0, 8):
        render(step, 0)
        time.sleep(0.7)
else:
    render(dx, dy)

st.caption("The chart is the whole 10,000-image test set, each digit moved up, down, left and right by the same "
           "amount, averaged. Neither network saw shifted digits in training.")
st.info(f"At 3 px the MLP gets {stats['shift_curve']['mlp'][3]:.0%} of test digits right; the CNN "
        f"{stats['shift_curve']['cnn'][3]:.0%}. The CNN is more tolerant, not immune: pooling only forgives small "
        "moves. In practice you also train on shifted, rotated and scaled copies (data augmentation).")

show_code(shift, note="Shifting just moves the pixel grid; the networks never saw these moved digits in training.")
