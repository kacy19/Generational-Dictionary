
"""
utils/constants.py — Generational Dictionary
Shared constants: generation definitions, column mappings, thresholds,
abbreviation expansions, and stop-word list.
"""

# ── Generation definitions ────────────────────────────────────────────────

GENERATIONS = ["Gen Alpha", "Gen Z", "Millennial", "Gen X", "Boomer"]

GENERATION_COLORS = {
    "Gen Alpha": "#00BBF9",
    "Gen Z":     "#9B5DE5",
    "Millennial":"#F15BB5",
    "Gen X":     "#FEE440",
    "Boomer":    "#00F5D4",
}

GENERATION_EMOJIS = {
    "Gen Alpha": "🤖",
    "Gen Z":     "✨",
    "Millennial":"🥑",
    "Gen X":     "📼",
    "Boomer":    "☎️",
}

GENERATION_YEARS = {
    "Gen Alpha": "2013–present",
    "Gen Z":     "1997–2012",
    "Millennial":"1981–1996",
    "Gen X":     "1965–1980",
    "Boomer":    "1946–1964",
}

GENERATION_TAGLINES = {
    "Gen Alpha": "Digital natives from birth",
    "Gen Z":     "No cap, it's bussin'",
    "Millennial":"LOL, adulting is hard",
    "Gen X":     "Whatever, as if",
    "Boomer":    "Back in my day…",
}

# Maps generation label → DataFrame column name
GENERATION_MEANING_COLS = {
    "Gen Alpha": "Gen Alpha Meaning",
    "Gen Z":     "Gen Z Meaning",
    "Millennial":"Millennial Meaning",
    "Gen X":     "Gen X Meaning",
    "Boomer":    "Boomer Meaning",
}

# ── Matching thresholds ───────────────────────────────────────────────────

EXACT_THRESHOLD    = 1.0   # cosine sim for confirmed exact match
SEMANTIC_THRESHOLD = 0.45  # minimum cosine sim to accept a semantic result

# ── Abbreviation expansion table ──────────────────────────────────────────

ABBREVIATIONS: dict[str, str] = {
    "idk":  "i do not know",
    "imo":  "in my opinion",
    "imho": "in my humble opinion",
    "tbh":  "to be honest",
    "ngl":  "not gonna lie",
    "irl":  "in real life",
    "afaik":"as far as i know",
    "lol":  "laughing out loud",
    "lmao": "laughing my head off",
    "omg":  "oh my god",
    "wtf":  "what the heck",
    "smh":  "shaking my head",
    "brb":  "be right back",
    "gg":   "good game",
    "gl":   "good luck",
    "ikr":  "i know right",
    "frfr": "for real for real",
    "fr":   "for real",
    "lmk":  "let me know",
    "hmu":  "hit me up",
    "wyd":  "what are you doing",
    "istg": "i swear to god",
    "ong":  "on god",
    "lowkey":"somewhat secretly",
    "highkey":"very obviously",
}

# ── Stop words (functional words to strip before matching) ────────────────

STOP_WORDS: set[str] = {
    "a", "an", "the", "is", "it", "in", "on", "at", "to", "for",
    "of", "and", "or", "but", "so", "as", "by", "with", "from",
    "that", "this", "these", "those", "be", "are", "was", "were",
    "i", "me", "my", "you", "your", "we", "our", "they", "their",
}
