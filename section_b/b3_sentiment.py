"""
b3_sentiment.py
───────────────
Hindi Sentiment Classifier:
  1. Generates balanced positive/negative/neutral Hindi sentences via LLM.
  2. Fine-tunes IndicBERT (ai4bharat/indic-bert) as a 3-class classifier.
  3. Exposes `predict(text)` for inference.

Usage:
    python section_b/b3_sentiment.py --generate --count 300
    python section_b/b3_sentiment.py --train
    python section_b/b3_sentiment.py --predict "यह फिल्म बहुत बढ़िया थी"
"""

import json
import sys
from pathlib import Path
import click
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from utils.ollama_client import ollama_chat

SYNTH_DATA_PATH = config.DATA_DIR / "sentiment_data.jsonl"
MODEL_OUTPUT_DIR = config.MODELS_DIR / "sentiment"
LABEL_MAP = {"positive": 0, "negative": 1, "neutral": 2}
ID2LABEL = {0: "positive", 1: "negative", 2: "neutral"}

TOPICS = ["खेल", "राजनीति", "खाना", "परिवार", "पढ़ाई", "मौसम", "स्वास्थ्य", "यात्रा"]


def generate_synthetic_data(count_per_class: int = 100) -> None:
    """Ask LLM to generate labelled Hindi sentences for each sentiment."""
    data: list[dict] = []
    for label in ["positive", "negative", "neutral"]:
        label_hi = {"positive": "सकारात्मक", "negative": "नकारात्मक", "neutral": "तटस्थ"}[label]
        for topic in tqdm(TOPICS, desc=f"Generating {label}"):
            n = max(1, count_per_class // len(TOPICS))
            prompt = (
                f"हिन्दी में '{topic}' विषय पर {n} {label_hi} भावना वाले वाक्य लिखें। "
                f"प्रत्येक वाक्य नई लाइन पर।"
            )
            response = ollama_chat(prompt)
            for line in response.splitlines():
                line = line.strip()
                if len(line) > 8:
                    data.append({"text": line, "label": label, "label_id": LABEL_MAP[label]})

    SYNTH_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SYNTH_DATA_PATH, "w", encoding="utf-8") as f:
        for d in data:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"✅ Synthetic sentiment data saved → {SYNTH_DATA_PATH}  ({len(data)} samples)")


def train_model() -> None:
    """Fine-tune IndicBERT for 3-class sentiment classification."""
    try:
        from transformers import (
            AutoTokenizer, AutoModelForSequenceClassification,
            TrainingArguments, Trainer,
        )
        from datasets import Dataset
        import numpy as np

    except ImportError:
        print("❌ Install: pip install transformers datasets")
        return

    if not SYNTH_DATA_PATH.exists():
        print("❌ No data. Run with --generate first.")
        return

    records = [json.loads(l) for l in SYNTH_DATA_PATH.read_text(encoding="utf-8").splitlines() if l]
    tokenizer = AutoTokenizer.from_pretrained(config.INDICBERT_MODEL)

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=config.MAX_SEQ_LENGTH,
            padding="max_length",
        )

    ds = Dataset.from_list(records).rename_column("label_id", "labels")
    ds = ds.map(tokenize, batched=True)
    split = ds.train_test_split(test_size=0.15, seed=42)

    model = AutoModelForSequenceClassification.from_pretrained(
        config.INDICBERT_MODEL, num_labels=3,
        id2label=ID2LABEL, label2id=LABEL_MAP,
    )
    args = TrainingArguments(
        output_dir=str(MODEL_OUTPUT_DIR),
        num_train_epochs=config.NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=config.BATCH_SIZE,
        learning_rate=config.LEARNING_RATE,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )
    Trainer(
        model=model, args=args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
    ).train()
    model.save_pretrained(MODEL_OUTPUT_DIR)
    tokenizer.save_pretrained(MODEL_OUTPUT_DIR)
    print(f"✅ Sentiment model saved → {MODEL_OUTPUT_DIR}")


def predict(text: str) -> dict:
    """Return sentiment label and confidence score."""
    if MODEL_OUTPUT_DIR.exists():
        from transformers import pipeline
        clf = pipeline("text-classification", model=str(MODEL_OUTPUT_DIR))
        result = clf(text)[0]
        return {"label": result["label"], "score": round(result["score"], 4)}

    # LLM fallback
    prompt = (
        f"इस हिन्दी वाक्य की भावना बताएं: '{text}'\n"
        "केवल एक शब्द: positive, negative, या neutral"
    )
    label = ollama_chat(prompt).strip().lower()
    if label not in LABEL_MAP:
        label = "neutral"
    return {"label": label, "score": 1.0, "source": "llm_fallback"}


@click.command()
@click.option("--generate", is_flag=True)
@click.option("--count", default=100, show_default=True, help="Samples per class")
@click.option("--train", "do_train", is_flag=True)
@click.option("--predict", "pred_text", default=None)
def main(generate: bool, count: int, do_train: bool, pred_text: str | None):
    if generate:
        generate_synthetic_data(count)
    if do_train:
        train_model()
    if pred_text:
        result = predict(pred_text)
        print(f"Sentiment: {result['label']} (confidence: {result.get('score', '?')})")


if __name__ == "__main__":
    main()
