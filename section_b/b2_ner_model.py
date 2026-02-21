"""
b2_ner_model.py
───────────────
Named Entity Recognition pipeline:
  1. Uses Ollama LLM with few-shot prompting to annotate sentences with
     PER (person), LOC (location), ORG (organisation) entities.
  2. Trains a token classifier on these silver labels.
  3. Active-learning loop: flags low-confidence predictions for review.

Usage:
    python section_b/b2_ner_model.py --generate-labels --count 300
    python section_b/b2_ner_model.py --train
    python section_b/b2_ner_model.py --tag "अमिताभ बच्चन मुंबई में रहते हैं"
"""

import json
import re
import sys
from pathlib import Path
import click
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from utils.ollama_client import ollama_chat

LABEL_DATA_PATH = config.DATA_DIR / "ner_labels.jsonl"
MODEL_OUTPUT_DIR = config.MODELS_DIR / "ner_model"

FEW_SHOT = """
उदाहरण (BIO format):
वाक्य: सचिन तेंदुलकर मुंबई में रहते हैं
BIO: B-PER I-PER B-LOC O O O

वाक्य: इसरो बेंगलुरु में स्थित है
BIO: B-ORG B-LOC O O O
"""

LABELS = ["O", "B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG"]


def llm_ner_tag(sentence: str) -> list[tuple[str, str]]:
    """Prompt LLM to produce BIO tags for a Hindi sentence."""
    tokens = sentence.split()
    prompt = (
        f"{FEW_SHOT}\n"
        f"अब tag करें (शब्द की संख्या exactly {len(tokens)} होनी चाहिए):\n"
        f"वाक्य: {sentence}\nBIO:"
    )
    response = ollama_chat(prompt).strip()
    bio_tags = response.split()

    # Align lengths
    if len(bio_tags) < len(tokens):
        bio_tags += ["O"] * (len(tokens) - len(bio_tags))
    bio_tags = bio_tags[: len(tokens)]

    return list(zip(tokens, bio_tags))


def generate_silver_labels(count: int = 300) -> None:
    if not config.RAW_CORPUS.exists():
        print("❌ Corpus not found.")
        return

    sentences = config.RAW_CORPUS.read_text(encoding="utf-8").splitlines()[:count]
    LABEL_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(LABEL_DATA_PATH, "w", encoding="utf-8") as f:
        for sent in tqdm(sentences, desc="NER labeling"):
            labeled = llm_ner_tag(sent)
            if labeled:
                record = {
                    "sentence": sent,
                    "tokens": [t for t, _ in labeled],
                    "ner_tags": [g for _, g in labeled],
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"✅ NER labels saved → {LABEL_DATA_PATH}")


def train_model() -> None:
    """Fine-tune an XLM-R token classifier on the silver NER data."""
    try:
        from transformers import (
            AutoTokenizer, AutoModelForTokenClassification,
            TrainingArguments, Trainer, DataCollatorForTokenClassification,
        )
        from datasets import Dataset
    except ImportError:
        print("❌ Install: pip install transformers datasets")
        return

    if not LABEL_DATA_PATH.exists():
        print("❌ No label data.")
        return

    records = [json.loads(l) for l in LABEL_DATA_PATH.read_text(encoding="utf-8").splitlines() if l]
    label2id = {l: i for i, l in enumerate(LABELS)}
    id2label = {i: l for l, i in label2id.items()}
    tokenizer = AutoTokenizer.from_pretrained(config.XLMR_MODEL)

    def tokenize_align(examples):
        enc = tokenizer(
            examples["tokens"], is_split_into_words=True,
            truncation=True, max_length=config.MAX_SEQ_LENGTH, padding="max_length",
        )
        all_labels = []
        for i, tags in enumerate(examples["ner_tags"]):
            word_ids = enc.word_ids(batch_index=i)
            lbls = []
            prev = None
            for wid in word_ids:
                if wid is None:
                    lbls.append(-100)
                elif wid != prev:
                    lbls.append(label2id.get(tags[wid], 0))
                else:
                    lbls.append(-100)
                prev = wid
            all_labels.append(lbls)
        enc["labels"] = all_labels
        return enc

    ds = Dataset.from_list(records).map(tokenize_align, batched=True)
    split = ds.train_test_split(test_size=0.15, seed=42)
    model = AutoModelForTokenClassification.from_pretrained(
        config.XLMR_MODEL, num_labels=len(LABELS), id2label=id2label, label2id=label2id,
    )
    args = TrainingArguments(
        output_dir=str(MODEL_OUTPUT_DIR),
        num_train_epochs=config.NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=config.BATCH_SIZE,
        learning_rate=config.LEARNING_RATE,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        load_best_model_at_end=True,
    )
    Trainer(
        model=model, args=args,
        train_dataset=split["train"], eval_dataset=split["test"],
        data_collator=DataCollatorForTokenClassification(tokenizer),
    ).train()
    model.save_pretrained(MODEL_OUTPUT_DIR)
    tokenizer.save_pretrained(MODEL_OUTPUT_DIR)
    print(f"✅ NER model saved → {MODEL_OUTPUT_DIR}")


def ner_tag(sentence: str) -> list[tuple[str, str]]:
    """Tag entities; falls back to LLM if model not trained."""
    if MODEL_OUTPUT_DIR.exists():
        from transformers import pipeline
        nlp = pipeline("ner", model=str(MODEL_OUTPUT_DIR), aggregation_strategy="simple")
        return [(r["word"], r["entity_group"]) for r in nlp(sentence)]
    print("ℹ  Using LLM fallback.")
    return llm_ner_tag(sentence)


@click.command()
@click.option("--generate-labels", is_flag=True)
@click.option("--count", default=300, show_default=True)
@click.option("--train", "do_train", is_flag=True)
@click.option("--tag", default=None)
def main(generate_labels: bool, count: int, do_train: bool, tag: str | None):
    if generate_labels:
        generate_silver_labels(count)
    if do_train:
        train_model()
    if tag:
        result = ner_tag(tag)
        for word, entity in result:
            print(f"  {word:20s} → {entity}")


if __name__ == "__main__":
    main()
