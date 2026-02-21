"""
c8_lang_detector.py
────────────────────
Hindi/English/Mixed Language Detector.
Wraps b5_lang_identifier to detect language per-word and route
to the appropriate processing model.

Usage:
    python section_c/c8_lang_detector.py --text "मैं office जा रहा हूँ"
    streamlit run section_c/c8_lang_detector.py
"""

import sys
from pathlib import Path
import click

sys.path.insert(0, str(Path(__file__).parent.parent))


def detect(text: str) -> dict:
    """Detect language using b5 classifier."""
    from section_b.b5_lang_identifier import detect as b5_detect
    return b5_detect(text)


def route_to_model(result: dict) -> str:
    """Decide which downstream model to use based on detected language."""
    label = result["label"]
    if label == "HI":
        return "Hindi NLP pipeline (POS, NER, Sentiment)"
    elif label == "EN":
        return "English NLP pipeline (spaCy / transformers-en)"
    else:
        return "Code-Mixed pipeline (character-level hybrid)"


@click.command()
@click.option("--text", default=None)
@click.option("--ui", is_flag=True, help="Launch Streamlit UI instead")
def cli(text: str | None, ui: bool):
    if ui:
        import subprocess
        subprocess.run(["streamlit", "run", __file__, "--", "--ui=false"])
        return
    if text:
        result = detect(text)
        print(f"Language: {result['label']} ({result['confidence']:.1%})")
        print(f"Routing:  {route_to_model(result)}")
        print("Word-level:")
        for word, lang in result["word_labels"]:
            print(f"  {word:20s} → {lang}")


# ── Streamlit UI ───────────────────────────────────────────────────────────────
def _streamlit_app():
    import streamlit as st

    st.set_page_config(page_title="हिन्दी Language Detector", page_icon="🔍")
    st.title("🔍 Language Detector – Hindi / English / Code-Mixed")

    input_text = st.text_area(
        "वाक्य डालें:",
        value="मैं tomorrow office जाऊँगा क्योंकि meeting है।",
        height=100,
    )
    LANG_COLORS = {"HI": "🟢", "EN": "🔵", "MIX": "🟡", "O": "⚪"}

    if st.button("🔍 Detect Language", type="primary"):
        with st.spinner("Detecting…"):
            result = detect(input_text)

        label = result["label"]
        conf = result["confidence"]

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Detected Language", f"{LANG_COLORS.get(label, '')} {label}")
            st.metric("Confidence", f"{conf:.1%}")
        with col2:
            st.info(f"**Routing:** {route_to_model(result)}")

        st.markdown("### Word-level Detection")
        wl_html = ""
        for word, lang in result["word_labels"]:
            color = {"HI": "#4CAF50", "EN": "#2196F3", "MIX": "#FF9800", "O": "#9E9E9E"}.get(lang, "#EEE")
            wl_html += (
                f'<span style="background:{color};color:white;padding:3px 7px;'
                f'border-radius:4px;margin:2px;display:inline-block">'
                f'{word} <sub>{lang}</sub></span> '
            )
        st.markdown(wl_html, unsafe_allow_html=True)


if __name__ == "__main__":
    import sys as _sys
    if "streamlit" in _sys.modules or "STREAMLIT_SERVER_PORT" in __import__("os").environ:
        _streamlit_app()
    else:
        cli()
