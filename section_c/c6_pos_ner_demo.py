"""
c6_pos_ner_demo.py
──────────────────
Interactive Streamlit app for Hindi POS Tagging & NER.
  - Text input → color-coded POS and NER annotations
  - LLM explains each tag in Hindi
  - Feedback collection for model improvement

Run:
    streamlit run section_c/c6_pos_ner_demo.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="हिन्दी POS & NER Demo",
    page_icon="🏷️",
    layout="wide",
)

POS_COLORS = {
    "NN": "#4CAF50", "VB": "#2196F3", "JJ": "#FF9800",
    "RB": "#9C27B0", "PR": "#F44336", "CC": "#009688",
    "PP": "#795548", "DT": "#607D8B", "QT": "#E91E63",
    "RP": "#3F51B5", "PU": "#9E9E9E",
}
NER_COLORS = {
    "B-PER": "#E53935", "I-PER": "#EF9A9A",
    "B-LOC": "#1E88E5", "I-LOC": "#90CAF9",
    "B-ORG": "#43A047", "I-ORG": "#A5D6A7",
    "O": "transparent",
}

TAG_DESCRIPTION = {
    "NN": "संज्ञा (Noun)", "VB": "क्रिया (Verb)", "JJ": "विशेषण (Adjective)",
    "RB": "क्रिया-विशेषण (Adverb)", "PR": "सर्वनाम (Pronoun)",
    "CC": "संयोजन (Conjunction)", "PP": "परसर्ग (Postposition)",
    "DT": "निर्धारक (Determiner)", "QT": "परिमाणवाचक (Quantifier)",
    "PER": "व्यक्ति (Person)", "LOC": "स्थान (Location)", "ORG": "संगठन (Organisation)",
}


@st.cache_resource
def load_pos():
    from section_b.b1_pos_tagger import tag_sentence
    return tag_sentence


@st.cache_resource
def load_ner():
    from section_b.b2_ner_model import ner_tag
    return ner_tag


def render_colored_tokens(tokens_tags: list[tuple[str, str]], color_map: dict) -> str:
    html = ""
    for word, tag in tokens_tags:
        color = color_map.get(tag, "#EEEEEE")
        clean_tag = tag.replace("B-", "").replace("I-", "")
        html += (
            f'<span style="background:{color};color:white;padding:3px 6px;'
            f'border-radius:4px;margin:2px;display:inline-block;font-size:14px;" '
            f'title="{TAG_DESCRIPTION.get(clean_tag, tag)}">'
            f'{word} <sub style="font-size:9px">{tag}</sub></span> '
        )
    return html


# ── UI ─────────────────────────────────────────────────────────────────────────
st.title("🏷️ हिन्दी POS & NER Demo")
st.markdown("**हिन्दी वाक्य डालें** – Part-of-Speech tags और Named Entities देखें")

sample_sentences = [
    "राम ने दिल्ली में एक अच्छी नौकरी पाई।",
    "प्रधानमंत्री नरेंद्र मोदी ने लाल किले पर भाषण दिया।",
    "भारतीय अंतरिक्ष अनुसंधान संगठन ने सफलतापूर्वक उपग्रह लॉन्च किया।",
]

col1, col2 = st.columns([3, 1])
with col1:
    input_text = st.text_area(
        "वाक्य यहाँ लिखें:", value=sample_sentences[0], height=100
    )
with col2:
    st.markdown("**नमूना वाक्य:**")
    for i, s in enumerate(sample_sentences):
        if st.button(f"उदाहरण {i+1}", key=f"sample_{i}"):
            input_text = s

if st.button("🔍 विश्लेषण करें", type="primary"):
    if input_text.strip():
        with st.spinner("विश्लेषण हो रहा है…"):
            tab1, tab2 = st.tabs(["🟢 POS Tagging", "🔵 Named Entities"])

            with tab1:
                st.markdown("### भाषाई भूमिकाएँ (POS Tags)")
                try:
                    pos_result = load_pos()(input_text)
                    st.markdown(
                        render_colored_tokens(pos_result, POS_COLORS),
                        unsafe_allow_html=True,
                    )
                    st.markdown("---")
                    st.markdown("**Legend:**")
                    legend_html = " ".join(
                        f'<span style="background:{c};color:white;padding:2px 5px;border-radius:3px;margin:2px;font-size:12px">{tag}: {TAG_DESCRIPTION.get(tag,tag)}</span>'
                        for tag, c in POS_COLORS.items()
                    )
                    st.markdown(legend_html, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"POS tagging failed: {e}")

            with tab2:
                st.markdown("### नामित इकाइयाँ (Named Entities)")
                try:
                    ner_result = load_ner()(input_text)
                    st.markdown(
                        render_colored_tokens(ner_result, NER_COLORS),
                        unsafe_allow_html=True,
                    )
                    entities = [(w, t) for w, t in ner_result if t != "O"]
                    if entities:
                        st.markdown("**पाई गई entities:**")
                        for word, tag in entities:
                            clean_tag = tag.replace("B-", "").replace("I-", "")
                            st.write(f"  • **{word}** → {TAG_DESCRIPTION.get(clean_tag, tag)}")
                    else:
                        st.info("कोई named entity नहीं मिली।")
                except Exception as e:
                    st.error(f"NER failed: {e}")

        # Feedback
        st.markdown("---")
        st.markdown("### 📝 फीडबैक")
        feedback = st.text_area("यदि कोई गलती हो तो सही annotation यहाँ लिखें:")
        if st.button("फीडबैक सबमिट करें"):
            fb_file = Path("data/feedback.jsonl")
            fb_file.parent.mkdir(exist_ok=True)
            import json
            with open(fb_file, "a", encoding="utf-8") as f:
                f.write(json.dumps({"text": input_text, "feedback": feedback}, ensure_ascii=False) + "\n")
            st.success("धन्यवाद! आपका feedback सहेज लिया गया है।")
    else:
        st.warning("कृपया पहले एक वाक्य डालें।")
