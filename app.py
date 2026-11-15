"""Neural Network Playground: the live demos for the ACT Neural Networks lecture.

    streamlit run app.py
"""
import streamlit as st

st.set_page_config(page_title="Neural Network Playground", page_icon="🧠", layout="wide")

pages = [
    st.Page("pages/home.py", title="Start here", icon="🏠", default=True),
    st.Page("pages/1_draw_a_digit.py", title="1 · Draw a digit", icon="✏️"),
    st.Page("pages/2_hidden_neurons.py", title="2 · What the neurons look for", icon="🔍"),
    st.Page("pages/3_shift_test.py", title="3 · The shift test", icon="↔️"),
    st.Page("pages/4_inside_the_cnn.py", title="4 · Inside the CNN", icon="🧱"),
]
st.navigation(pages).run()
