"""
utils/helpers.py — Generational Dictionary
Utility functions for display: confidence labels, generation card info,
and text truncation.
"""

from utils.constants import (
    GENERATION_COLORS, GENERATION_EMOJIS,
    GENERATION_YEARS, GENERATION_TAGLINES,
    SEMANTIC_THRESHOLD,
)


def confidence_label(score: float) -> tuple[str, str]:
    """
    Map a cosine similarity score to a human-readable label and colour.

    Returns (label, hex_colour).
    """
    if score >= 0.90:
        return "High",   "#00F5D4"
    if score >= 0.75:
        return "Good",   "#9B5DE5"
    if score >= SEMANTIC_THRESHOLD:
        return "Low",    "#FEE440"
    return     "None",   "#888888"


def generation_card_info(gen: str) -> dict:
    """Return display metadata for a single generation card."""
    return {
        "color":   GENERATION_COLORS.get(gen, "#888888"),
        "emoji":   GENERATION_EMOJIS.get(gen, ""),
        "years":   GENERATION_YEARS.get(gen, ""),
        "tagline": GENERATION_TAGLINES.get(gen, ""),
    }


def truncate(text: str, max_chars: int = 120) -> str:
    """Truncate text to max_chars, appending ellipsis if needed."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "…"
