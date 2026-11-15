"""Shared Streamlit pieces: page style, the digit picker and the 'show the code' box."""
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import streamlit as st

from .models import CNN, MLP, MODEL_DIR
from .preprocess import canvas_to_mnist

CSS = """
<style>
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
code, pre { font-family: 'IBM Plex Mono', monospace !important; }
.status { font-size: 1.15rem; background: #fffdf8; border-left: 6px solid #1f6b66;
          padding: 0.6rem 1rem; border-radius: 6px; margin: 0.3rem 0 0.8rem 0; }
.kicker { font-family: 'IBM Plex Mono', monospace; color: #b04618; letter-spacing: 1px;
          text-transform: uppercase; font-size: 0.85rem; margin-bottom: -0.6rem; }
</style>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@400;600&display=swap" rel="stylesheet">
"""


def page_header(kicker, title, blurb):
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(f'<p class="kicker">{kicker}</p>', unsafe_allow_html=True)
    st.title(title)
    st.markdown(blurb, unsafe_allow_html=True)


def status(text):
    st.markdown(f'<div class="status">{text}</div>', unsafe_allow_html=True)


@st.cache_resource
def load_models():
    return MLP(), CNN()


@st.cache_data
def load_examples():
    ex = np.load(MODEL_DIR / "examples.npz")
    return ex["images"].astype(np.float32) / 255.0, ex["labels"]


@st.cache_data
def load_stats():
    return json.loads((MODEL_DIR / "stats.json").read_text())


def _set_digit(img, source, label=None):
    st.session_state["digit"] = img
    st.session_state["digit_source"] = source
    st.session_state["digit_label"] = label


def current_digit():
    if "digit" not in st.session_state:
        images, labels = load_examples()
        _set_digit(images[7], "MNIST test example", int(labels[7]))
    return st.session_state["digit"]


def digit_picker(key, canvas_size=260):
    """Draw a digit, or pick a real MNIST test digit. The choice is shared between pages."""
    from streamlit_drawable_canvas import st_canvas

    current_digit()
    images, labels = load_examples()
    st.markdown("**Draw a digit** (big, in the middle of the box), or pick a real one from the test set:")
    cols = st.columns(10)
    for d in range(10):
        if cols[d].button(str(d), key=f"{key}-ex-{d}", use_container_width=True):
            _set_digit(images[d], "MNIST test example", int(labels[d]))

    resets = st.session_state.setdefault("canvas_resets", {})
    result = st_canvas(stroke_width=20, stroke_color="#FFFFFF", background_color="#000000",
                       height=canvas_size, width=canvas_size, drawing_mode="freedraw",
                       update_streamlit=True, return_image_data=True,
                       key=f"{key}-canvas-{resets.get(key, 0)}")
    if result.image_data is not None:
        h = hashlib.sha1(result.image_data.tobytes()).hexdigest()
        seen = st.session_state.setdefault("canvas_seen", {})
        if seen.get(key) != h:
            seen[key] = h
            img = canvas_to_mnist(result.image_data)
            if img is not None:
                _set_digit(img, "your drawing")
    if st.button("Clear the drawing", key=f"{key}-clear"):
        resets[key] = resets.get(key, 0) + 1
        st.rerun()
    return st.session_state["digit"]


def describe_source():
    src = st.session_state.get("digit_source", "")
    label = st.session_state.get("digit_label")
    return f"{src} (true label {label})" if label is not None else src


def show_code(*objects, note=None):
    with st.expander("Show the code"):
        if note:
            st.markdown(note)
        for obj in objects:
            st.code(inspect.getsource(obj), language="python")


def repo_path(*parts):
    return Path(__file__).resolve().parent.parent.joinpath(*parts)
