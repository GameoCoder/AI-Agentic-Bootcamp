"""
d2_corpus_stats_dashboard.py
────────────────────────────
Hindi Corpus Statistics Dashboard (Streamlit + Plotly).
  - Token counts, vocabulary size, sentence count
  - Top-N unigram / bigram / trigram bar charts
  - Vocabulary growth curve
  - Natural language query agent (LLM → corpus context)

Run:
    streamlit run section_d/d2_corpus_stats_dashboard.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from utils.ollama_client import ollama_chat


# ── Pure-Python Data Loaders (importable without Streamlit) ────────────────────

def load_unigrams() -> dict:
    if config.UNIGRAMS_JSON.exists():
        return json.loads(config.UNIGRAMS_JSON.read_text(encoding="utf-8"))
    return {}


def load_bigrams() -> dict:
    if config.BIGRAMS_JSON.exists():
        return json.loads(config.BIGRAMS_JSON.read_text(encoding="utf-8"))
    return {}


def load_trigrams() -> dict:
    if config.TRIGRAMS_JSON.exists():
        return json.loads(config.TRIGRAMS_JSON.read_text(encoding="utf-8"))
    return {}


def corpus_basic_stats() -> dict:
    stats: dict = {}
    if config.RAW_CORPUS.exists():
        lines = config.RAW_CORPUS.read_text(encoding="utf-8").splitlines()
        stats["sentences"] = len(lines)
        stats["total_tokens"] = sum(len(l.split()) for l in lines)
    if config.TOKENIZED_CORPUS.exists():
        tok_lines = config.TOKENIZED_CORPUS.read_text(encoding="utf-8").splitlines()
        stats["tokenized_sentences"] = len(tok_lines)
    return stats


def vocab_growth_data(sample_n: int = 200):
    """Compute vocabulary growth over first sample_n sentences. Returns list of dicts."""
    import pandas as pd
    if not config.TOKENIZED_CORPUS.exists():
        return pd.DataFrame()
    lines = config.TOKENIZED_CORPUS.read_text(encoding="utf-8").splitlines()
    step = max(1, len(lines) // sample_n)
    vocab: set[str] = set()
    records = []
    for i, line in enumerate(lines[::step], 1):
        vocab.update(line.split())
        records.append({"Sentences (×step)": i * step, "Vocabulary Size": len(vocab)})
    return pd.DataFrame(records)


# ── Streamlit UI (guarded – safe to import in tests) ───────────────────────────

def _run_streamlit_app():
    import streamlit as st
    import plotly.express as px
    import pandas as pd

    st.set_page_config(page_title="हिन्दी Corpus Statistics", page_icon="📈", layout="wide")
    st.title("📈 हिन्दी Corpus Statistics Dashboard")

    unigrams = load_unigrams()
    bigrams  = load_bigrams()
    trigrams = load_trigrams()
    basic    = corpus_basic_stats()

    # === Summary Metrics ===
    st.markdown("## 📋 Corpus Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("वाक्य (Sentences)", f"{basic.get('sentences', 0):,}")
    c2.metric("कुल Tokens", f"{basic.get('total_tokens', 0):,}")
    c3.metric("Vocabulary Size", f"{len(unigrams):,}")
    c4.metric("Bigrams", f"{len(bigrams):,}")

    st.divider()

    # === N-gram Charts ===
    top_n = st.slider("Top-N N-grams दिखाएं:", 5, 50, 20)
    tab1, tab2, tab3 = st.tabs(["Unigrams", "Bigrams", "Trigrams"])

    with tab1:
        if unigrams:
            data = pd.DataFrame(list(unigrams.items())[:top_n], columns=["Word", "Count"])
            fig = px.bar(data, x="Count", y="Word", orientation="h",
                         title=f"Top-{top_n} Unigrams", color="Count",
                         color_continuous_scale="Blues")
            fig.update_layout(yaxis=dict(autorange="reversed"), height=500)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Unigram data not found. Run section_a/a3_unigram_freq.py first.")

    with tab2:
        if bigrams:
            data = pd.DataFrame(list(bigrams.items())[:top_n], columns=["Bigram", "Count"])
            fig = px.bar(data, x="Count", y="Bigram", orientation="h",
                         title=f"Top-{top_n} Bigrams", color="Count",
                         color_continuous_scale="Greens")
            fig.update_layout(yaxis=dict(autorange="reversed"), height=500)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Bigram data not found. Run section_a/a4_bigram_freq.py first.")

    with tab3:
        if trigrams:
            data = pd.DataFrame(list(trigrams.items())[:top_n], columns=["Trigram", "Count"])
            fig = px.bar(data, x="Count", y="Trigram", orientation="h",
                         title=f"Top-{top_n} Trigrams", color="Count",
                         color_continuous_scale="Oranges")
            fig.update_layout(yaxis=dict(autorange="reversed"), height=500)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Trigram data not found. Run section_a/a5_trigram_freq.py first.")

    st.divider()

    # === Vocabulary Growth ===
    st.markdown("## 📈 Vocabulary Growth Curve")
    growth_df = vocab_growth_data()
    if not growth_df.empty:
        import plotly.express as pxp
        fig = pxp.line(growth_df, x="Sentences (×step)", y="Vocabulary Size",
                      title="Heaps' Law – Vocabulary Growth",
                      line_shape="spline", color_discrete_sequence=["#7C4DFF"])
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Vocabulary growth data unavailable. Run A1–A3 pipeline first.")

    st.divider()

    # === NL Query Agent ===
    st.markdown("## 🤖 Natural Language Query Agent")
    st.caption("Corpus के बारे में कुछ भी पूछें (हिन्दी या English में)")

    query = st.text_input("प्रश्न:", placeholder="जैसे: सबसे सामान्य 5 शब्द कौन से हैं?")

    if st.button("🔍 पूछें", key="nl_query") and query:
        context_parts = []
        if unigrams:
            context_parts.append(f"Top-5 unigrams: {list(unigrams.items())[:5]}")
            context_parts.append(f"Vocabulary size: {len(unigrams)}")
        if basic:
            context_parts.append(f"Corpus stats: {basic}")
        if bigrams:
            context_parts.append(f"Top-3 bigrams: {list(bigrams.items())[:3]}")

        context_str = "\n".join(context_parts) if context_parts else "No corpus data loaded yet."
        prompt = (
            f"Hindi corpus के बारे में यह जानकारी उपलब्ध है:\n{context_str}\n\n"
            f"प्रश्न: {query}\n\nहिन्दी में उत्तर दें।"
        )
        with st.spinner("उत्तर खोजा जा रहा है…"):
            ans = ollama_chat(prompt)
        st.markdown(f"**उत्तर:** {ans}")


# Guard: only run UI when executed via `streamlit run`
_IS_STREAMLIT = "streamlit" in __import__("sys").modules or \
                __import__("os").environ.get("STREAMLIT_SERVER_PORT") is not None
if _IS_STREAMLIT:
    _run_streamlit_app()
