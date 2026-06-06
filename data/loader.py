
"""
data/loader.py — Generational Dictionary
Dataset loading, caching, and statistics utilities.
"""

import pandas as pd
import streamlit as st
from pathlib import Path

DATA_PATH = Path(__file__).parent.parent / "data" / "generational_slang.xlsx"


@st.cache_data(show_spinner=False)
def load_dataset(path: str | Path = DATA_PATH) -> pd.DataFrame:
    """Load and validate the generational slang dataset from Excel."""
    df = pd.read_excel(path)
    required = {"Text", "Label", "Category",
                "Gen Z Meaning", "Millennial Meaning",
                "Gen X Meaning", "Boomer Meaning", "Gen Alpha Meaning"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing columns: {missing}")
    df = df.dropna(subset=["Text"]).reset_index(drop=True)
    df["Text"] = df["Text"].str.strip().str.lower()
    return df


def dataset_stats(df: pd.DataFrame) -> dict:
    """Return summary statistics for sidebar display."""
    return {
        "total": len(df),
        "categories": df["Category"].nunique(),
        "generations": df["Label"].nunique(),
        "gen_counts": df["Label"].value_counts().to_dict(),
    }


def get_category_list(df: pd.DataFrame) -> list[str]:
    """Return sorted list of unique categories, with All at front."""
    cats = sorted(df["Category"].dropna().unique().tolist())
    return ["All"] + cats

