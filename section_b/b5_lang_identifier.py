"""
b5_lang_identifier.py
─────────────────────
Language Identification for Code-Mixed Hindi-English:
  1. Generates synthetic code-mixed sentences using Ollama LLM.
  2. Trains a character n-gram + SGD classifier (scikit-learn).
  3. Detects language switches at word level.

Labels: HI (Hindi), EN (English), MIX (code-mixed)

Usage:
    python section_b/b5_lang_identifier.py --generate --count 500
    python section_b/b5_lang_identifier.py --train
    python section_b/b5_lang_identifier.py --detect "मैं office जा रहा हूँ"
"""

import json
import pickle
import re
import sys
from pathlib import Path
import click
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from utils.ollama_client import ollama_chat

SYNTH_DATA_PATH = config.DATA_DIR / "lang_id_data.jsonl"
MODEL_PATH = config.MODELS_DIR / "lang_identifier.pkl"

HINDI_RE = re.compile(r"[\u0900-\u097F]")
ENGLISH_RE = re.compile(r"[a-zA-Z]")


def word_lang(word: str) -> str:
    hi = sum(1 for c in word if HINDI_RE.match(c))
    en = sum(1 for c in word if ENGLISH_RE.match(c))
    total = len(word)
    if total == 0:
        return "O"
    if hi / total > 0.5:
        return "HI"
    if en / total > 0.5:
        return "EN"
    return "MIX"


def generate_codemixed_data(count: int = 500) -> None:
    data: list[dict] = []

    # Code-mixed
    for _ in tqdm(range(count // 3), desc="Generating code-mixed"):
        prompt = "एक वाक्य लिखें जिसमें हिन्दी और English दोनों शब्द मिले हों (code-mixing)।"
        sent = ollama_chat(prompt).strip()
        if sent:
            data.append({"text": sent, "label": "MIX"})

    # Pure Hindi (from corpus)
    if config.RAW_CORPUS.exists():
        lines = config.RAW_CORPUS.read_text(encoding="utf-8").splitlines()
        for line in lines[: count // 3]:
            data.append({"text": line.strip(), "label": "HI"})

    # English (prompt LLM)
    for _ in range(count // 3):
        prompt = "Write a short English sentence about India or daily life."
        sent = ollama_chat(prompt, system="You are a helpful assistant. Reply in English only.").strip()
        if sent:
            data.append({"text": sent, "label": "EN"})

    SYNTH_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SYNTH_DATA_PATH, "w", encoding="utf-8") as f:
        for d in data:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"✅ Lang-ID data saved → {SYNTH_DATA_PATH}  ({len(data)} samples)")


def train_model() -> None:
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import SGDClassifier
    from sklearn.feature_extraction.text import TfidfVectorizer

    if not SYNTH_DATA_PATH.exists():
        print("❌ No data. Run --generate first.")
        return

    records = [json.loads(l) for l in SYNTH_DATA_PATH.read_text(encoding="utf-8").splitlines() if l]
    X = [r["text"] for r in records]
    y = [r["label"] for r in records]

    model = Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), max_features=50000)),
        ("clf", SGDClassifier(loss="modified_huber", max_iter=1000, random_state=42)),
    ])
    model.fit(X, y)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"✅ Language identifier saved → {MODEL_PATH}")
    accuracy = model.score(X, y)
    print(f"   Training accuracy: {accuracy:.2%}")


def detect(text: str) -> dict:
    """Return sentence-level label and word-level labels."""
    word_labels = [(w, word_lang(w)) for w in text.split()]

    if MODEL_PATH.exists():
        with open(MODEL_PATH, "rb") as f:
            clf = pickle.load(f)
        sentence_label = clf.predict([text])[0]
        proba = clf.predict_proba([text])[0]
        confidence = float(proba.max())
    else:
        hi_count = sum(1 for _, l in word_labels if l == "HI")
        en_count = sum(1 for _, l in word_labels if l == "EN")
        total = len(word_labels)
        if hi_count / total > 0.7:
            sentence_label, confidence = "HI", 0.9
        elif en_count / total > 0.7:
            sentence_label, confidence = "EN", 0.9
        else:
            sentence_label, confidence = "MIX", 0.7

    return {
        "sentence": text,
        "label": sentence_label,
        "confidence": round(confidence, 4),
        "word_labels": word_labels,
    }


@click.command()
@click.option("--generate", is_flag=True)
@click.option("--count", default=500, show_default=True)
@click.option("--train", "do_train", is_flag=True)
@click.option("--detect", "det_text", default=None)
def main(generate: bool, count: int, do_train: bool, det_text: str | None):
    if generate:
        generate_codemixed_data(count)
    if do_train:
        train_model()
    if det_text:
        result = detect(det_text)
        print(f"Label: {result['label']} ({result['confidence']:.1%})")
        print("Word-level:", result["word_labels"])


if __name__ == "__main__":
    main()
