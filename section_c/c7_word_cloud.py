"""
c7_word_cloud.py
────────────────
Hindi Word Cloud Generator (Streamlit app).
  - Upload text or paste content
  - Removes stop words from data/stopwords.txt
  - Generates word cloud image
  - LLM suggests context-specific stop words to remove

Run:
    streamlit run section_c/c7_word_cloud.py
"""

import sys
import json
import io
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import config
from utils.ollama_client import ollama_chat

st.set_page_config(page_title="हिन्दी Word Cloud", page_icon="☁️", layout="wide")

HINDI_RE = re.compile(r"[\u0900-\u097F]")


@st.cache_data
def load_stopwords() -> set[str]:
    if config.STOPWORDS_TXT.exists():
        return set(config.STOPWORDS_TXT.read_text(encoding="utf-8").splitlines())
    # Minimal built-in
    return {"का", "की", "के", "में", "है", "हैं", "को", "से", "पर", "और", "यह", "वह", "एक"}


def tokenize(text: str) -> list[str]:
    try:
        from indicnlp.tokenize import indic_tokenize
        return indic_tokenize.trivial_tokenize(text, lang="hi")
    except ImportError:
        return text.split()


def compute_freq(text: str, stopwords: set[str], extra_stops: set[str]) -> dict[str, int]:
    from collections import Counter
    all_stops = stopwords | extra_stops
    tokens = tokenize(text)
    filtered = [
        t for t in tokens
        if t not in all_stops and len(t) > 1 and any(HINDI_RE.match(c) for c in t)
    ]
    return dict(Counter(filtered).most_common(200))


def generate_wordcloud(freq: dict[str, int]) -> bytes:
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt

    wc = WordCloud(
        font_path=None,        # system default; user may need to specify a Devanagari font
        width=900, height=500,
        background_color="white",
        colormap="viridis",
        max_words=150,
        prefer_horizontal=0.8,
    ).generate_from_frequencies(freq)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def llm_suggest_extra_stops(text: str, top_words: list[str]) -> list[str]:
    words_str = ", ".join(top_words[:30])
    prompt = (
        f"इस हिन्दी पाठ के संदर्भ में इन शब्दों में से कौन से word cloud के लिए "
        f"अनुपयोगी/सामान्य शब्द हैं?\n\nशब्द: {words_str}\n\n"
        "केवल हटाने योग्य शब्दों की comma-separated सूची दें (अधिकतम 10)।"
    )
    response = ollama_chat(prompt)
    return [w.strip() for w in response.split(",") if w.strip()]


# ── UI ─────────────────────────────────────────────────────────────────────────
st.title("☁️ हिन्दी Word Cloud Generator")

stopwords = load_stopwords()

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### पाठ डालें")
    uploaded = st.file_uploader("फ़ाइल अपलोड करें (.txt)", type=["txt"])
    if uploaded:
        text_input = uploaded.read().decode("utf-8")
        st.text_area("पाठ (पूर्वावलोकन):", text_input[:400] + "…", height=120)
    else:
        text_input = st.text_area(
            "यहाँ हिन्दी पाठ पेस्ट करें:",
            value="भारत एक विविधताओं वाला देश है। यहाँ अनेक भाषाएँ, संस्कृतियाँ और धर्म हैं।",
            height=180,
        )

    extra_stop_input = st.text_input("अतिरिक्त stop words (comma-separated):", "")
    extra_stops = {w.strip() for w in extra_stop_input.split(",") if w.strip()}

    use_llm = st.checkbox("🤖 LLM से stop words सुझाव लें", value=False)
    generate_btn = st.button("☁️ Word Cloud बनाएं", type="primary")

with col2:
    if generate_btn and text_input.strip():
        with st.spinner("Word cloud बना रहा है…"):
            freq = compute_freq(text_input, stopwords, extra_stops)
            if not freq:
                st.warning("कोई शब्द नहीं मिले। stop words कम करें।")
            else:
                if use_llm:
                    top_words = list(freq.keys())[:40]
                    llm_stops = llm_suggest_extra_stops(text_input, top_words)
                    st.info(f"LLM ने सुझाए stop words: {', '.join(llm_stops)}")
                    freq = {k: v for k, v in freq.items() if k not in llm_stops}

                img_bytes = generate_wordcloud(freq)
                st.image(img_bytes, caption="हिन्दी Word Cloud", use_container_width=True)

                # Frequency table
                st.markdown("**शीर्ष शब्द:**")
                st.dataframe(
                    {"शब्द": list(freq.keys())[:20], "आवृत्ति": list(freq.values())[:20]},
                    hide_index=True,
                )
