"""
b4_text_classifier.py
─────────────────────
Topic Text Classifier:
  1. Uses Ollama zero-shot prompting to label raw corpus sentences by topic.
  2. Trains a multilingual DistilBERT on the resulting silver-labeled data.
  3. Iterative active-learning: flags low-confidence predictions.

Topics: news, sports, entertainment, politics, science, health, education, technology

Usage:
    python section_b/b4_text_classifier.py --label --count 500
    python section_b/b4_text_classifier.py --train
    python section_b/b4_text_classifier.py --classify "सरकार ने नई शिक्षा नीति लागू की"
"""

import json
import sys
from pathlib import Path
import click
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from utils.ollama_client import ollama_chat

LABEL_DATA_PATH = config.DATA_DIR / "topic_labels.jsonl"
MODEL_OUTPUT_DIR = config.MODELS_DIR / "text_classifier"
LOW_CONF_PATH = config.DATA_DIR / "low_confidence_topics.jsonl"

TOPICS = ["news", "sports", "entertainment", "politics", "science", "health", "education", "technology"]
TOPICS_HI = {
    "news": "समाचार", "sports": "खेल", "entertainment": "मनोरंजन",
    "politics": "राजनीति", "science": "विज्ञान", "health": "स्वास्थ्य",
    "education": "शिक्षा", "technology": "प्रौद्योगिकी"
}
LABEL_MAP = {t: i for i, t in enumerate(TOPICS)}
ID2LABEL = {i: t for t, i in LABEL_MAP.items()}


def llm_classify(sentence: str) -> tuple[str, float]:
    """Zero-shot classify a sentence into a topic; return (topic, confidence)."""
    topics_str = ", ".join(TOPICS)
    prompt = (
        f"इस हिन्दी वाक्य का विषय क्या है?\nवाक्य: {sentence}\n"
        f"विकल्प: {topics_str}\n"
        "केवल एक शब्द (English) उत्तर दें।"
    )
    response = ollama_chat(prompt).strip().lower()
    for topic in TOPICS:
        if topic in response:
            return topic, 0.85
    return "news", 0.3  # low confidence default


def generate_labels(count: int = 500) -> None:
    if not config.RAW_CORPUS.exists():
        print("❌ Corpus not found.")
        return

    sentences = config.RAW_CORPUS.read_text(encoding="utf-8").splitlines()[:count]
    LABEL_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOW_CONF_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(LABEL_DATA_PATH, "w", encoding="utf-8") as f, \
         open(LOW_CONF_PATH, "w", encoding="utf-8") as lf:
        for sent in tqdm(sentences, desc="Topic labeling"):
            topic, conf = llm_classify(sent)
            record = {"text": sent, "label": topic, "label_id": LABEL_MAP[topic], "confidence": conf}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            if conf < 0.5:
                lf.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"✅ Topic labels saved → {LABEL_DATA_PATH}")


def train_model() -> None:
    try:
        from transformers import (
            AutoTokenizer, AutoModelForSequenceClassification,
            TrainingArguments, Trainer,
        )
        from datasets import Dataset
    except ImportError:
        print("❌ Install transformers + datasets")
        return

    if not LABEL_DATA_PATH.exists():
        print("❌ No data. Run --label first.")
        return

    records = [json.loads(l) for l in LABEL_DATA_PATH.read_text(encoding="utf-8").splitlines() if l]
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-multilingual-cased")

    def tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=128, padding="max_length")

    ds = Dataset.from_list(records).rename_column("label_id", "labels").map(tok, batched=True)
    split = ds.train_test_split(test_size=0.15, seed=42)

    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-multilingual-cased", num_labels=len(TOPICS),
        id2label=ID2LABEL, label2id=LABEL_MAP,
    )
    args = TrainingArguments(
        output_dir=str(MODEL_OUTPUT_DIR),
        num_train_epochs=config.NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=config.BATCH_SIZE,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )
    Trainer(model=model, args=args,
            train_dataset=split["train"], eval_dataset=split["test"]).train()
    model.save_pretrained(MODEL_OUTPUT_DIR)
    tokenizer.save_pretrained(MODEL_OUTPUT_DIR)
    print(f"✅ Text classifier saved → {MODEL_OUTPUT_DIR}")


def classify(text: str) -> dict:
    if MODEL_OUTPUT_DIR.exists():
        from transformers import pipeline
        clf = pipeline("text-classification", model=str(MODEL_OUTPUT_DIR))
        r = clf(text)[0]
        return {"label": r["label"], "score": round(r["score"], 4)}
    topic, conf = llm_classify(text)
    return {"label": topic, "score": conf, "source": "llm"}


@click.command()
@click.option("--label", is_flag=True)
@click.option("--count", default=500, show_default=True)
@click.option("--train", "do_train", is_flag=True)
@click.option("--classify", "clf_text", default=None)
def main(label: bool, count: int, do_train: bool, clf_text: str | None):
    if label:
        generate_labels(count)
    if do_train:
        train_model()
    if clf_text:
        r = classify(clf_text)
        print(f"Topic: {r['label']}  (confidence: {r['score']})")


if __name__ == "__main__":
    main()
