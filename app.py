"""
app.py — Generational Dictionary
Streamlit UI: translate slang and expressions across generations.
"""

import streamlit as st
import pandas as pd

# ── Page config (must be first Streamlit call) ──────────────────────────────
st.set_page_config(
    page_title="Generational Dictionary",
    page_icon="🗣️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Local imports ────────────────────────────────────────────────────────────
from data.loader import load_dataset, dataset_stats, get_category_list
from models.model import load_model
from models.embeddings import get_dataset_embeddings
from core.translator import translate
from utils.constants import (
    APP_TITLE, APP_SUBTITLE,
    GENERATIONS, GENERATION_EMOJIS, GENERATION_COLORS,
    GENERATION_YEARS, GENERATION_TAGLINES,
)
from utils.helpers import confidence_label, generation_card_info, truncate

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global ── */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    color: #f0f0f0;
}
[data-testid="stSidebar"] {
    background: rgba(15,12,41,0.95) !important;
    border-right: 1px solid rgba(255,255,255,0.08);
}
h1, h2, h3 { color: #ffffff; }
.stTextInput > div > div > input {
    background: rgba(255,255,255,0.08) !important;
    color: #fff !important;
    border: 1px solid rgba(255,255,255,0.2) !important;
    border-radius: 12px !important;
    font-size: 1.1rem !important;
    padding: 0.6rem 1rem !important;
}

/* ── Generation Cards ── */
.gen-card {
    border-radius: 16px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.6rem;
    border: 1px solid rgba(255,255,255,0.10);
    backdrop-filter: blur(6px);
    transition: transform 0.15s;
}
.gen-card:hover { transform: translateY(-2px); }
.gen-card-header {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 0.5rem;
}
.gen-badge {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    padding: 2px 8px;
    border-radius: 999px;
    text-transform: uppercase;
}
.gen-translation {
    font-size: 1.05rem;
    font-weight: 600;
    color: #ffffff;
    line-height: 1.4;
}
.gen-tagline {
    font-size: 0.75rem;
    opacity: 0.6;
    margin-top: 0.3rem;
}
.gen-years {
    font-size: 0.7rem;
    opacity: 0.5;
    margin-top: 0.15rem;
}

/* ── Meaning Box ── */
.meaning-box {
    background: rgba(255,255,255,0.06);
    border-left: 4px solid #9B5DE5;
    border-radius: 0 12px 12px 0;
    padding: 1rem 1.2rem;
    margin: 1rem 0;
    font-size: 0.95rem;
    color: #ddd;
    line-height: 1.6;
}

/* ── Confidence pill ── */
.conf-pill {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 999px;
    font-size: 0.8rem;
    font-weight: 600;
}

/* ── Similar entries ── */
.similar-item {
    background: rgba(255,255,255,0.05);
    border-radius: 8px;
    padding: 0.5rem 0.8rem;
    margin-bottom: 0.4rem;
    font-size: 0.88rem;
    color: #ccc;
}

/* ── Hero banner ── */
.hero-banner {
    background: linear-gradient(90deg,#9B5DE5,#F15BB5,#00BBF9);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.4rem;
    font-weight: 900;
    letter-spacing: -0.02em;
    margin-bottom: 0;
    line-height: 1.1;
}
.hero-sub {
    color: rgba(255,255,255,0.55);
    font-size: 1rem;
    margin-top: 0.3rem;
    margin-bottom: 1.5rem;
}

/* ── Stat pills in sidebar ── */
.stat-pill {
    background: rgba(255,255,255,0.07);
    border-radius: 8px;
    padding: 0.4rem 0.7rem;
    margin-bottom: 0.35rem;
    font-size: 0.82rem;
    color: #ddd;
    display: flex;
    justify-content: space-between;
}

/* ── Tab styling ── */
[data-testid="stTabs"] button {
    color: rgba(255,255,255,0.6) !important;
    font-weight: 600;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #fff !important;
    border-bottom-color: #9B5DE5 !important;
}

/* ── Unknown warning ── */
.unknown-warn {
    background: rgba(255,200,50,0.10);
    border: 1px solid rgba(255,200,50,0.3);
    border-radius: 12px;
    padding: 0.8rem 1rem;
    color: #ffe082;
    font-size: 0.9rem;
    margin-bottom: 1rem;
}

/* ── Origin chip ── */
.origin-chip {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 700;
    margin-right: 6px;
}

/* ── Examples grid ── */
.example-btn-wrapper button {
    background: rgba(255,255,255,0.07) !important;
    color: #ddd !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 8px !important;
    font-size: 0.82rem !important;
}
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# Initialise resources (cached)
# ════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def init_resources():
    """Load dataset, model, embeddings once per server session."""
    df         = load_dataset()
    model      = load_model()
    embeddings = get_dataset_embeddings(df, model)
    return df, model, embeddings

with st.spinner("🧠 Loading Generational Dictionary…"):
    df, model, embeddings = init_resources()

stats      = dataset_stats(df)
categories = get_category_list(df)

# ════════════════════════════════════════════════════════════════
# Sidebar
# ════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🗣️ Gen Dictionary")
    st.caption("Bridge language gaps across generations")

    st.divider()

    # Dataset stats
    st.markdown("**📊 Dataset**")
    stat_labels = {
        "Total entries":   "🗂️",
        "Gen Z":           "✨",
        "Millennial":      "☕",
        "Gen X":           "😎",
        "Boomer":          "📺",
        "Categories":      "🏷️",
    }
    for k, v in stats.items():
        emoji = stat_labels.get(k, "•")
        st.markdown(
            f'<div class="stat-pill"><span>{emoji} {k}</span><span><b>{v:,}</b></span></div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # Filter by category
    st.markdown("**🏷️ Filter by Category**")
    selected_cat = st.selectbox(
        "Category",
        ["All"] + categories,
        label_visibility="collapsed",
    )

    st.divider()

    # Generation legend
    st.markdown("**🌐 Generations**")
    for gen in GENERATIONS:
        info = generation_card_info(gen)
        st.markdown(
            f"{info['emoji']} **{gen}** · _{info['years']}_  \n"
            f"<span style='font-size:0.75rem;opacity:0.6'>{info['tagline']}</span>",
            unsafe_allow_html=True,
        )
        st.markdown("")

    st.divider()
    st.caption("Built with ❤️ using SentenceTransformers + Streamlit")

# ════════════════════════════════════════════════════════════════
# Main content
# ════════════════════════════════════════════════════════════════

st.markdown('<p class="hero-banner">🗣️ Generational Dictionary</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-sub">Translate slang & expressions across Gen Alpha · Gen Z · Millennials · Gen X · Boomers</p>',
    unsafe_allow_html=True,
)

# ── Quick-example chips ──────────────────────────────────────────
EXAMPLES = [
    "no cap", "bussin", "snatched", "bet", "lowkey",
    "rizz", "slay", "it's giving", "that's fire", "vibe check",
    "sus", "bestie", "drip", "goated", "tea",
    "why are you so burnout", "full drip mode", "snatched tho fr",
]

st.markdown("**💡 Try an example:**")
cols = st.columns(6)
for i, ex in enumerate(EXAMPLES[:12]):
    with cols[i % 6]:
        if st.button(ex, key=f"ex_{i}", use_container_width=True):
            st.session_state["query_input"] = ex

st.markdown("---")

# ── Search input ─────────────────────────────────────────────────
col_input, col_btn = st.columns([5, 1])
with col_input:
    query = st.text_input(
        "Enter a word, phrase, or sentence:",
        placeholder="e.g.  no cap  /  snatched  /  that's lowkey bussin fr",
        key="query_input",
        label_visibility="collapsed",
    )
with col_btn:
    translate_btn = st.button("🔍 Translate", type="primary", use_container_width=True)

# ────────────────────────────────────────────────────────────────
# Run translation
# ────────────────────────────────────────────────────────────────

if query and (translate_btn or query):

    # Apply category filter to df
    working_df = df.copy()
    if selected_cat != "All":
        filtered = df[df["Category"] == selected_cat]
        working_df = filtered if len(filtered) > 0 else df

    with st.spinner("🔄 Translating across generations…"):
        result = translate(query, working_df, embeddings, model)

    # ── Meta row ────────────────────────────────────────────────
    st.markdown("")
    meta_cols = st.columns([3, 2, 2, 2])

    with meta_cols[0]:
        matched = result["matched_text"] or query
        st.markdown(f"**🎯 Matched:** `{truncate(matched, 60)}`")

    with meta_cols[1]:
        gen   = result["detected_gen"]
        color = GENERATION_COLORS.get(gen, "#888")
        emoji = GENERATION_EMOJIS.get(gen, "🌐")
        st.markdown(
            f'**Origin:** <span class="origin-chip" style="background:{color}22;color:{color};border:1px solid {color}44">'
            f'{emoji} {gen}</span>',
            unsafe_allow_html=True,
        )

    with meta_cols[2]:
        cat = result["category"]
        st.markdown(f"**Category:** `{cat}`")

    with meta_cols[3]:
        conf_str = result["confidence_label"]
        pct = int(result["confidence"] * 100)
        st.markdown(f"**Confidence:** {conf_str} ({pct}%)")

    st.markdown("")

    # ── Unknown warning ─────────────────────────────────────────
    if result["is_unknown"]:
        st.markdown(
            f'<div class="unknown-warn">⚠️ <b>"{query}"</b> was not found in the dictionary. '
            f'Showing as-is. The expression may be too new or too specific.</div>',
            unsafe_allow_html=True,
        )

    # ── Meaning box ─────────────────────────────────────────────
    st.markdown(
        f'<div class="meaning-box">📖 {result["meaning"]}</div>',
        unsafe_allow_html=True,
    )

    # ── Translation cards ────────────────────────────────────────
    st.markdown("### 🌐 Translations Across Generations")

    # Two rows × 3 columns (5 gens + 1 spacer)
    card_order = ["Gen Alpha", "Gen Z", "Millennial", "Gen X", "Boomer"]
    row1 = card_order[:3]
    row2 = card_order[3:]

    def render_card(gen: str, translation: str):
        info  = generation_card_info(gen)
        color = info["color"]
        emoji = info["emoji"]
        years = info["years"]
        tag   = info["tagline"]
        is_origin = (gen == result["detected_gen"])
        border_style = f"border: 2px solid {color};" if is_origin else ""
        origin_badge = f'<span class="gen-badge" style="background:{color}33;color:{color};">🎯 Origin</span>' if is_origin else ""
        st.markdown(
            f"""<div class="gen-card" style="background: {color}18; {border_style}">
                <div class="gen-card-header">
                    <span style="font-size:1.4rem">{emoji}</span>
                    <span style="font-weight:700;font-size:1rem;color:#fff">{gen}</span>
                    {origin_badge}
                </div>
                <div class="gen-translation">{translation}</div>
                <div class="gen-tagline">{tag}</div>
                <div class="gen-years">{years}</div>
            </div>""",
            unsafe_allow_html=True,
        )
    cols1 = st.columns(3)
    for i, gen in enumerate(row1):
        with cols1[i]:
            render_card(gen, result["translations"].get(gen, query))

    cols2 = st.columns([1, 1, 1])
    for i, gen in enumerate(row2):
        with cols2[i]:
            render_card(gen, result["translations"].get(gen, query))

    # ── Tabs: similar entries + explore ─────────────────────────
    st.markdown("")
    tab1, tab2 = st.tabs(["🔗 Related Expressions", "📋 Full Table View"])

    with tab1:
        similar = result.get("similar", [])
        if similar:
            for item in similar:
                gen_color = GENERATION_COLORS.get(item["gen"], "#888")
                gen_emoji = GENERATION_EMOJIS.get(item["gen"], "")
                score_pct = int(item["score"] * 100)
                st.markdown(
                    f'<div class="similar-item">'
                    f'<span style="color:{gen_color};font-weight:700">{gen_emoji} {item["gen"]}</span> · '
                    f'<b>{item["text"]}</b> · '
                    f'<span style="opacity:0.6">{item["cat"]}</span> · '
                    f'<span style="opacity:0.5">{score_pct}% match</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.info("No related expressions found.")

    with tab2:
        # Show a clean DataFrame of the translations
        table_data = {
            "Generation": [],
            "Born":        [],
            "Translation": [],
        }
        for gen in card_order:
            info = generation_card_info(gen)
            table_data["Generation"].append(f"{info['emoji']} {gen}")
            table_data["Born"].append(info["years"])
            table_data["Translation"].append(result["translations"].get(gen, query))

        tdf = pd.DataFrame(table_data)
        st.dataframe(
            tdf,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Generation": st.column_config.TextColumn("Generation", width=130),
                "Born":        st.column_config.TextColumn("Era", width=110),
                "Translation": st.column_config.TextColumn("Translation"),
            },
        )

        # Download
        csv = tdf.to_csv(index=False).encode()
        st.download_button(
            "⬇️ Download results as CSV",
            data=csv,
            file_name=f"gendictionary_{query[:30].replace(' ','_')}.csv",
            mime="text/csv",
        )

else:
    # Landing state: show generation overview cards
    st.markdown("### 🌐 Meet the Generations")
    landing_cols = st.columns(5)
    for i, gen in enumerate(GENERATIONS):
        info = generation_card_info(gen)
        with landing_cols[i]:
            st.markdown(
                f"""<div class="gen-card" style="background:{info['color']}18;
                    border:1px solid {info['color']}33; text-align:center; padding:1.4rem 0.8rem;">
                    <div style="font-size:2rem">{info['emoji']}</div>
                    <div style="font-weight:700;font-size:1rem;color:#fff;margin:0.4rem 0">{gen}</div>
                    <div style="font-size:0.72rem;opacity:0.6;margin-bottom:0.5rem">{info['years']}</div>
                    <div style="font-size:0.78rem;opacity:0.75;line-height:1.4">{info['tagline']}</div>
                </div>""",
                unsafe_allow_html=True,
            )

    st.markdown("")
    st.info(
        "👆 Type a word or phrase above — or click one of the example chips — "
        "to see how each generation would say it."
    )

    # Show random sample from dataset
    st.markdown("### 🎲 Sample Entries from the Dictionary")
    sample = df.sample(10, random_state=42)[
        ["Text", "Label", "Category", "Gen Z Meaning", "Millennial Meaning", "Boomer Meaning"]
    ].rename(columns={
        "Text": "Expression",
        "Label": "Origin",
        "Gen Z Meaning": "Gen Z",
        "Millennial Meaning": "Millennial",
        "Boomer Meaning": "Boomer",
    })
    st.dataframe(sample, use_container_width=True, hide_index=True)


