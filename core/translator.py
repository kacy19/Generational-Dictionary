
"""
core/translator.py — Generational Dictionary
Hybrid two-phase matching: exact lexical lookup then SBERT semantic fallback.
"""

import re
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from utils.constants import (
    GENERATIONS, GENERATION_MEANING_COLS,
    ABBREVIATIONS, STOP_WORDS,
    EXACT_THRESHOLD, SEMANTIC_THRESHOLD,
)


# ── Pre-processing ────────────────────────────────────────────────────────

def preprocess(text: str) -> str:
    """Lowercase, expand abbreviations, strip punctuation, remove stop-words."""
    text = text.lower().strip()
    tokens = text.split()
    tokens = [ABBREVIATIONS.get(t, t) for t in tokens]
    tokens = [re.sub(r"[^a-z0-9\s]", "", t) for t in tokens]
    tokens = [t for t in tokens if t and t not in STOP_WORDS]
    return " ".join(tokens)


# ── Phase 1: exact lookup ─────────────────────────────────────────────────

def exact_lookup(
    query: str,
    df: pd.DataFrame,
) -> tuple[dict | None, float]:
    """
    Search the dataset for an exact string match.
    Returns the matched row as a dict and confidence 1.0, or (None, 0.0).
    """
    match = df[df["Text"] == query]
    if not match.empty:
        row = match.iloc[0]
        return _build_result(row, 1.0), 1.0
    return None, 0.0


# ── Phase 2: semantic fallback ────────────────────────────────────────────

def semantic_lookup(
    query: str,
    df: pd.DataFrame,
    embeddings: np.ndarray,
    model: SentenceTransformer,
) -> tuple[dict | None, float]:
    """
    Encode the query and compute cosine similarity against the dataset.
    Returns the best match and its similarity score, or (None, 0.0) if below
    SEMANTIC_THRESHOLD.
    """
    q_emb = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )[0]
    scores = embeddings @ q_emb  # dot product of unit vectors = cosine sim
    best_idx = int(np.argmax(scores))
    best_score = float(scores[best_idx])
    if best_score < SEMANTIC_THRESHOLD:
        return None, best_score
    row = df.iloc[best_idx]
    return _build_result(row, best_score), best_score


# ── Public interface ──────────────────────────────────────────────────────

def translate(
    raw_query: str,
    df: pd.DataFrame,
    embeddings: np.ndarray,
    model: SentenceTransformer,
) -> dict:
    """
    Translate *raw_query* using the hybrid pipeline.

    Returns a dict with keys: matched_text, generation, category,
    confidence, explanation, translations, similar, phase.
    """
    processed = preprocess(raw_query)

    # Phase 1: exact lookup
    result, score = exact_lookup(processed, df)
    phase = "exact"

    # Phase 2: semantic fallback
    if result is None:
        result, score = semantic_lookup(processed, df, embeddings, model)
        phase = "semantic"

    if result is None:
        return _no_match(raw_query)

    result["confidence"] = score
    result["phase"] = phase
    result["similar"] = _find_similar(processed, df, embeddings, model, top_k=4)
    return result


# ── Helpers ───────────────────────────────────────────────────────────────

def _build_result(row: pd.Series, score: float) -> dict:
    translations = {
        gen: str(row.get(col, row["Text"]))
        for gen, col in GENERATION_MEANING_COLS.items()
    }
    return {
        "matched_text": row["Text"],
        "generation":   row.get("Label", "Unknown"),
        "category":     row.get("Category", "General"),
        "explanation":  row.get("Gen Z Meaning", ""),
        "translations": translations,
        "confidence":   score,
        "phase":        "exact",
        "similar":      [],
    }


def _find_similar(
    query: str,
    df: pd.DataFrame,
    embeddings: np.ndarray,
    model: SentenceTransformer,
    top_k: int = 4,
) -> list[dict]:
    """Return top_k semantically similar entries (excluding the top result)."""
    q_emb = model.encode(
        [query], convert_to_numpy=True, normalize_embeddings=True
    )[0]
    scores = embeddings @ q_emb
    top_indices = np.argsort(scores)[::-1][1: top_k + 1]
    return [
        {
            "text":  df.iloc[i]["Text"],
            "gen":   df.iloc[i].get("Label", ""),
            "cat":   df.iloc[i].get("Category", ""),
            "score": float(scores[i]),
        }
        for i in top_indices
        if float(scores[i]) > SEMANTIC_THRESHOLD
    ]


def _no_match(query: str) -> dict:
    return {
        "matched_text": query,
        "generation":   "Unknown",
        "category":     "Unknown",
        "explanation":  "No matching expression found in the dataset.",
        "translations": {gen: query for gen in GENERATIONS},
        "confidence":   0.0,
        "phase":        "none",
        "similar":      [],
    }

