

"""
models/model.py — Generational Dictionary
SentenceTransformer model loader with Streamlit caching.
"""

import streamlit as st
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"


@st.cache_resource(show_spinner=False)
def load_model(model_name: str = MODEL_NAME) -> SentenceTransformer:
    """Load the Sentence-BERT model once per server session."""
    return SentenceTransformer(model_name)

