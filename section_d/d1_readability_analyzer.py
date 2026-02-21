"""
d1_readability_analyzer.py
──────────────────────────
Hindi Text Readability Analyzer (Streamlit dashboard).
Computes:
  - Average sentence length
  - Average word length
  - OOV rate vs. unigram dictionary
  - Hindi Flesch-Kincaid adaptation (shorter sentences / simpler words = easier)
  - LLM vocabulary difficulty score (1-10)
Displays Plotly gauge charts and a detailed breakdown.

Run:
    streamlit run section_d/d1_readability_analyzer.py
"""

import re
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from utils.ollama_client import ollama_chat

try:
    from indicnlp.tokenize import indic_tokenize
    def _tokenize(s): return indic_tokenize.trivial_tokenize(s, lang="hi")
except ImportError:
    def _tokenize(s): return s.split()

HINDI_RE = re.compile(r"[\u0900-\u097F]")


def split_sentences(text: str) -> list[str]:
    """Split Hindi text into sentences on ।/./!/?"""
    parts = re.split(r"[।.!?]+", text)
    return [p.strip() for p in parts if p.strip()]


def compute_metrics(text: str, vocab: set[str]) -> dict:
    sentences = split_sentences(text)
    all_tokens: list[str] = []
    for s in sentences:
        all_tokens.extend(_tokenize(s))

    hindi_tokens = [t for t in all_tokens if any(HINDI_RE.match(c) for c in t)]
    if not hindi_tokens:
        return {}

    avg_sent_len = len(hindi_tokens) / max(len(sentences), 1)
    avg_word_len = sum(len(t) for t in hindi_tokens) / len(hindi_tokens)
    oov_rate = sum(1 for t in hindi_tokens if t not in vocab) / len(hindi_tokens)

    # Hindi FK adaptation: higher score = more readable (0–100)
    # Inspired by: Reading_Ease = 206.835 – 1.015*(words/sentences) – 84.6*(syllables/words)
    # For Hindi we approximate syllables ≈ character_count / 2.5
    avg_syllables = avg_word_len / 2.5
    fk_score = max(0, min(100, 206.835 - 1.015 * avg_sent_len - 84.6 * avg_syllables))

    return {
        "total_sentences": len(sentences),
        "total_words": len(hindi_tokens),
        "avg_sentence_length": round(avg_sent_len, 1),
        "avg_word_length": round(avg_word_len, 2),
        "oov_rate": round(oov_rate * 100, 1),
        "fk_reading_ease": round(fk_score, 1),
    }


def llm_difficulty_score(text: str) -> dict:
    prompt = (
        f"इस हिन्दी पाठ की शब्दावली की कठिनाई 1-10 में बताएं "
        f"(1=बहुत सरल, 10=बहुत कठिन) और एक पंक्ति में कारण बताएं:\n\n"
        f"{text[:500]}\n\nFormat: score|reason"
    )
    response = ollama_chat(prompt)
    parts = response.split("|", 1)
    nums = re.findall(r"\d+", parts[0])
    score = int(nums[0]) if nums else 5
    reason = parts[1].strip() if len(parts) > 1 else response
    return {"llm_score": max(1, min(10, score)), "llm_reason": reason}


def load_vocab() -> set[str]:
    if config.UNIGRAMS_JSON.exists():
        data = json.loads(config.UNIGRAMS_JSON.read_text(encoding="utf-8"))
        return set(data.keys())
    return set()


# ── Streamlit UI (guarded – importable without display server) ─────────────────
def _run_streamlit_app():
    import streamlit as st
    import plotly.graph_objects as go

    st.set_page_config(page_title="हिन्दी Readability Analyzer", page_icon="📊", layout="wide")
    st.title("📊 हिन्दी Text Readability Analyzer")

    vocab = load_vocab()
    st.caption(f"📖 Dictionary: {len(vocab):,} words loaded" if vocab
               else "⚠ Dictionary not available – OOV rate will be 0%")

    sample_text = (
        "भारत एक विशाल देश है। यहाँ अनेक भाषाएँ बोली जाती हैं। "
        "हिन्दी सबसे अधिक बोली जाने वाली भाषा है। "
        "देश की अर्थव्यवस्था तेजी से विकसित हो रही है।"
    )
    text_input = st.text_area("हिन्दी पाठ डालें:", value=sample_text, height=180)
    use_llm = st.checkbox("🤖 LLM से कठिनाई स्कोर भी लें", value=True)

    if st.button("📊 विश्लेषण करें", type="primary"):
        with st.spinner("विश्लेषण हो रहा है…"):
            metrics = compute_metrics(text_input, vocab)
            if not metrics:
                st.error("पाठ में हिन्दी शब्द नहीं मिले।")
            else:
                llm_info = llm_difficulty_score(text_input) if use_llm else {}

                def gauge(title, value, max_val, color):
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number", value=value,
                        title={"text": title, "font": {"size": 14}},
                        gauge={"axis": {"range": [0, max_val]},
                               "bar": {"color": color}, "bgcolor": "white"},
                    ))
                    fig.update_layout(height=200, margin=dict(t=40, b=10, l=20, r=20))
                    return fig

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.plotly_chart(gauge("शब्द/वाक्य", metrics["avg_sentence_length"], 40, "#2196F3"), use_container_width=True)
                with col2:
                    st.plotly_chart(gauge("औसत शब्द-लंबाई", metrics["avg_word_length"], 12, "#4CAF50"), use_container_width=True)
                with col3:
                    st.plotly_chart(gauge("OOV दर %", metrics["oov_rate"], 100, "#FF9800"), use_container_width=True)
                with col4:
                    st.plotly_chart(gauge("पठनीयता (FK)", metrics["fk_reading_ease"], 100, "#9C27B0"), use_container_width=True)

                st.markdown("### 📋 विस्तृत आँकड़े")
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("कुल वाक्य", metrics["total_sentences"])
                    st.metric("कुल शब्द", metrics["total_words"])
                    st.metric("औसत वाक्य-लंबाई", f"{metrics['avg_sentence_length']} शब्द")
                with col_b:
                    st.metric("औसत शब्द-लंबाई", f"{metrics['avg_word_length']} अक्षर")
                    st.metric("OOV दर", f"{metrics['oov_rate']}%")
                    st.metric("FK पठनीयता", f"{metrics['fk_reading_ease']} / 100")

                if llm_info:
                    st.markdown("### 🤖 LLM कठिनाई मूल्यांकन")
                    st.metric("LLM कठिनाई स्कोर", f"{llm_info['llm_score']} / 10")
                    st.info(f"**कारण:** {llm_info['llm_reason']}")

                fk = metrics["fk_reading_ease"]
                verdict = (
                    "✅ सरल – प्राथमिक स्तर के पाठकों के लिए उपयुक्त" if fk >= 70
                    else "🟡 मध्यम – माध्यमिक स्तर के पाठकों के लिए उपयुक्त" if fk >= 50
                    else "🔴 कठिन – उच्च शिक्षित पाठकों के लिए"
                )
                st.success(f"**पठनीयता निर्णय:** {verdict}")


# Called by `streamlit run` which executes the module at top-level.
# Guard prevents execution during pytest imports.
_IS_STREAMLIT = "streamlit" in __import__("sys").modules or \
                __import__("os").environ.get("STREAMLIT_SERVER_PORT") is not None
if _IS_STREAMLIT:
    _run_streamlit_app()
