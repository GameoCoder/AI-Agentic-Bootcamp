"""
c3_summarizer.py
────────────────
Hindi Abstractive Text Summarizer.
Uses Ollama qwen2:0.5b for zero-shot / few-shot summarization.
Optionally fine-tunes google/mt5-small using teacher-generated summaries.

Usage:
    python section_c/c3_summarizer.py --text "लंबा हिन्दी लेख..."
    python section_c/c3_summarizer.py --file article.txt
    python section_c/c3_summarizer.py --generate-training --count 200
    python section_c/c3_summarizer.py --train
"""

import json
import sys
from pathlib import Path
import click
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from utils.ollama_client import ollama_chat

TRAIN_DATA_PATH = config.DATA_DIR / "summarizer_train.jsonl"
MODEL_OUTPUT_DIR = config.MODELS_DIR / "summarizer"


def summarize_with_llm(text: str, max_words: int = 50) -> str:
    """Generate an abstractive summary using Ollama."""
    prompt = (
        f"निम्नलिखित हिन्दी पाठ का संक्षेप (summary) लगभग {max_words} शब्दों में लिखें:\n\n"
        f"{text[:2000]}\n\n"
        f"संक्षेप:"
    )
    return ollama_chat(prompt)


def generate_training_pairs(count: int = 200) -> None:
    """Generate (article, summary) pairs for fine-tuning."""
    if not config.RAW_CORPUS.exists():
        print("❌ Corpus not found.")
        return

    lines = config.RAW_CORPUS.read_text(encoding="utf-8").splitlines()
    # Combine random consecutive lines as "articles"
    pairs: list[dict] = []
    step = 5
    for i in tqdm(range(0, min(count * step, len(lines) - step), step), desc="Generating summaries"):
        article = " ".join(lines[i : i + step])
        if len(article) < 100:
            continue
        summary = summarize_with_llm(article, max_words=30)
        pairs.append({"article": article, "summary": summary})
        if len(pairs) >= count:
            break

    TRAIN_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(TRAIN_DATA_PATH, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"✅ Training pairs saved → {TRAIN_DATA_PATH}  ({len(pairs)} pairs)")


def train_mt5() -> None:
    """Fine-tune mT5-small on generated summaries."""
    try:
        from transformers import (
            AutoTokenizer, AutoModelForSeq2SeqLM,
            Seq2SeqTrainingArguments, Seq2SeqTrainer,
            DataCollatorForSeq2Seq,
        )
        from datasets import Dataset
    except ImportError:
        print("❌ Install transformers + datasets")
        return

    if not TRAIN_DATA_PATH.exists():
        print("❌ No training data. Run --generate-training first.")
        return

    records = [json.loads(l) for l in TRAIN_DATA_PATH.read_text(encoding="utf-8").splitlines() if l]
    tokenizer = AutoTokenizer.from_pretrained(config.MT5_MODEL)

    def preprocess(batch):
        model_inputs = tokenizer(
            ["summarize: " + a for a in batch["article"]],
            max_length=512, truncation=True, padding="max_length",
        )
        labels = tokenizer(
            batch["summary"],
            max_length=128, truncation=True, padding="max_length",
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    ds = Dataset.from_list(records).map(preprocess, batched=True)
    split = ds.train_test_split(test_size=0.1, seed=42)
    model = AutoModelForSeq2SeqLM.from_pretrained(config.MT5_MODEL)

    args = Seq2SeqTrainingArguments(
        output_dir=str(MODEL_OUTPUT_DIR),
        num_train_epochs=config.NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=max(1, config.BATCH_SIZE // 2),
        predict_with_generate=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )
    Seq2SeqTrainer(
        model=model, args=args,
        train_dataset=split["train"], eval_dataset=split["test"],
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
    ).train()
    model.save_pretrained(MODEL_OUTPUT_DIR)
    tokenizer.save_pretrained(MODEL_OUTPUT_DIR)
    print(f"✅ Summarizer model saved → {MODEL_OUTPUT_DIR}")


def summarize(text: str) -> str:
    """Summarize using fine-tuned model or Ollama fallback."""
    if MODEL_OUTPUT_DIR.exists():
        from transformers import pipeline
        pipe = pipeline("summarization", model=str(MODEL_OUTPUT_DIR))
        result = pipe(text[:1024], max_length=100, min_length=20)
        return result[0]["summary_text"]
    return summarize_with_llm(text)


@click.command()
@click.option("--text", default=None)
@click.option("--file", "file_path", default=None, type=click.Path(exists=True))
@click.option("--generate-training", is_flag=True)
@click.option("--count", default=200, show_default=True)
@click.option("--train", "do_train", is_flag=True)
def main(text: str | None, file_path: str | None, generate_training: bool, count: int, do_train: bool):
    if generate_training:
        generate_training_pairs(count)
    if do_train:
        train_mt5()
    if text:
        print(summarize(text))
    if file_path:
        content = Path(file_path).read_text(encoding="utf-8")
        print(summarize(content))


if __name__ == "__main__":
    main()
