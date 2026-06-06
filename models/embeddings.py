
"""
models/embeddings.py — Generational Dictionary
Pre-compute and cache dataset-level sentence embeddings.
"""

import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer


@st.cache_data(show_spinner=False)
def get_dataset_embeddings(
    _df: pd.DataFrame,
    _model: SentenceTransformer,
) -> np.ndarray:
    """
    Encode every Text entry in the dataset into a dense embedding vector.
    Results are cached so encoding runs only once per session.
    """
    texts = _df["Text"].tolist()
    embeddings = _model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return embeddings

