"""
b1_pos_tagger.py
────────────────
Part-of-Speech Tagger:
  1. Uses Ollama (qwen2:0.5b) with few-shot prompting to generate
     silver POS labels for corpus sentences.
  2. Fine-tunes XLM-RoBERTa as a token classifier on these labels.
  3. Provides an inference function for tagging new sentences.

POS Tagset (simplified):
  NN=Noun, VB=Verb, JJ=Adjective, RB=Adverb,
  PR=Pronoun, CC=Conjunction, PP=Postposition,
  DT=Determiner, QT=Quantifier, RP=Particle, PU=Punctuation

Usage:
    python section_b/b1_pos_tagger.py --generate-labels --count 500
    python section_b/b1_pos_tagger.py --train
    python section_b/b1_pos_tagger.py --tag "राम ने खाना खाया"
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

LABEL_DATA_PATH = config.DATA_DIR / "pos_labels.jsonl"
MODEL_OUTPUT_DIR = config.MODELS_DIR / "pos_tagger"

FEW_SHOT_EXAMPLES = """
उदाहरण:
वाक्य: राम ने आम खाया
टैग: राम/NN ने/PP आम/NN खाया/VB

वाक्य: वह बहुत सुंदर लड़की है
टैग: वह/PR बहुत/RB सुंदर/JJ लड़की/NN है/VB
"""


def llm_pos_tag(sentence: str) -> list[tuple[str, str]]:
    """Ask LLM to POS-tag a sentence; returns [(word, tag)] pairs."""
    prompt = (
        f"{FEW_SHOT_EXAMPLES}\n"
        f"अब इस वाक्य को tag करें (format: word/TAG):\nवाक्य: {sentence}\nटैग:"
    )
    response = ollama_chat(prompt)
    tokens = response.strip().split()
    pairs = []
    for tok in tokens:
        if "/" in tok:
            parts = tok.rsplit("/", 1)
            pairs.append((parts[0], parts[1].upper()))
    return pairs if pairs else [(w, "NN") for w in sentence.split()]


def generate_silver_labels(count: int = 500) -> None:
    """Generate POS-labeled sentences using LLM and save to JSONL."""
    if not config.RAW_CORPUS.exists():
        print("❌ raw_corpus.txt not found. Run section_a/a1_corpus_crawler.py first.")
        return

    sentences = config.RAW_CORPUS.read_text(encoding="utf-8").splitlines()[:count]
    LABEL_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(LABEL_DATA_PATH, "w", encoding="utf-8") as f:
        for sent in tqdm(sentences, desc="POS tagging"):
            labeled = llm_pos_tag(sent)
            if labeled:
                record = {
                    "sentence": sent,
                    "tokens": [t for t, _ in labeled],
                    "tags": [t for _, t in labeled],
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"✅ Silver POS labels saved → {LABEL_DATA_PATH}")


def train_model() -> None:
    """Fine-tune XLM-R on generated silver labels."""
    try:
        from transformers import (
            AutoTokenizer, AutoModelForTokenClassification,
            TrainingArguments, Trainer, DataCollatorForTokenClassification,
        )
        from datasets import Dataset
        import numpy as np
    except ImportError:
        print("❌ transformers/datasets not installed. Run: pip install transformers datasets")
        return

    if not LABEL_DATA_PATH.exists():
        print("❌ No label data. Run with --generate-labels first.")
        return

    records = [json.loads(l) for l in LABEL_DATA_PATH.read_text(encoding="utf-8").splitlines() if l]

    all_tags = sorted({t for r in records for t in r["tags"]})
    tag2id = {t: i for i, t in enumerate(all_tags)}
    id2tag = {i: t for t, i in tag2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(config.XLMR_MODEL)

    def tokenize_and_align(examples):
        enc = tokenizer(
            examples["tokens"],
            is_split_into_words=True,
            truncation=True,
            max_length=config.MAX_SEQ_LENGTH,
            padding="max_length",
        )
        labels_out = []
        for i, tags in enumerate(examples["tags"]):
            word_ids = enc.word_ids(batch_index=i)
            label_ids = []
            prev = None
            for wid in word_ids:
                if wid is None:
                    label_ids.append(-100)
                elif wid != prev:
                    label_ids.append(tag2id.get(tags[wid], 0))
                else:
                    label_ids.append(-100)
                prev = wid
            labels_out.append(label_ids)
        enc["labels"] = labels_out
        return enc

    ds = Dataset.from_list(records).map(tokenize_and_align, batched=True)
    split = ds.train_test_split(test_size=0.1, seed=42)

    model = AutoModelForTokenClassification.from_pretrained(
        config.XLMR_MODEL, num_labels=len(tag2id), id2label=id2tag, label2id=tag2id
    )
    args = TrainingArguments(
        output_dir=str(MODEL_OUTPUT_DIR),
        num_train_epochs=config.NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=config.BATCH_SIZE,
        learning_rate=config.LEARNING_RATE,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        logging_dir=str(MODEL_OUTPUT_DIR / "logs"),
        load_best_model_at_end=True,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        data_collator=DataCollatorForTokenClassification(tokenizer),
    )
    trainer.train()
    model.save_pretrained(MODEL_OUTPUT_DIR)
    tokenizer.save_pretrained(MODEL_OUTPUT_DIR)
    print(f"✅ POS tagger saved → {MODEL_OUTPUT_DIR}")


def tag_sentence(sentence: str) -> list[tuple[str, str]]:
    """Tag a sentence using the fine-tuned model (falls back to LLM)."""
    if MODEL_OUTPUT_DIR.exists():
        from transformers import pipeline
        nlp = pipeline("token-classification", model=str(MODEL_OUTPUT_DIR), aggregation_strategy="simple")
        results = nlp(sentence)
        return [(r["word"], r["entity_group"]) for r in results]
    print("ℹ  Fine-tuned model not found, using LLM fallback.")
    return llm_pos_tag(sentence)


@click.command()
@click.option("--generate-labels", is_flag=True)
@click.option("--count", default=500, show_default=True)
@click.option("--train", "do_train", is_flag=True)
@click.option("--tag", default=None, help="Tag a single sentence")
def main(generate_labels: bool, count: int, do_train: bool, tag: str | None):
    if generate_labels:
        generate_silver_labels(count)
    if do_train:
        train_model()
    if tag:
        result = tag_sentence(tag)
        print("Tagged:")
        for word, pos in result:
            print(f"  {word:15s}  →  {pos}")


if __name__ == "__main__":
    main()
