"""
c9_transliteration_tool.py
──────────────────────────
Roman → Hindi Devanagari Transliteration Tool.
  - Primary: fine-tuned mT5-small model (if trained)
  - Fallback: Ollama LLM
  - Streamlit UI for batch and single conversion

Run:
    streamlit run section_c/c9_transliteration_tool.py
    python section_c/c9_transliteration_tool.py --train
    python section_c/c9_transliteration_tool.py --convert "computer"
"""

import json
import sys
from pathlib import Path
import click

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from utils.ollama_client import ollama_chat

MODEL_OUTPUT_DIR = config.MODELS_DIR / "transliterator"


def llm_transliterate(roman: str) -> str:
    """Ask Ollama LLM to transliterate Roman → Hindi Devanagari."""
    prompt = (
        f"Roman/English शब्द '{roman}' को हिन्दी Devanagari script में लिखें। "
        "केवल हिन्दी शब्द दें।"
    )
    return ollama_chat(prompt).strip()


def transliterate(roman: str) -> str:
    """Transliterate using fine-tuned model or LLM fallback."""
    if MODEL_OUTPUT_DIR.exists():
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        tok = AutoTokenizer.from_pretrained(str(MODEL_OUTPUT_DIR))
        model = AutoModelForSeq2SeqLM.from_pretrained(str(MODEL_OUTPUT_DIR))
        inputs = tok(f"transliterate: {roman}", return_tensors="pt")
        output = model.generate(**inputs, max_new_tokens=50)
        return tok.decode(output[0], skip_special_tokens=True)
    return llm_transliterate(roman)


def train_mt5() -> None:
    """Fine-tune mT5-small on transliteration pairs from a11."""
    try:
        from transformers import (
            AutoTokenizer, AutoModelForSeq2SeqLM,
            Seq2SeqTrainer, Seq2SeqTrainingArguments,
            DataCollatorForSeq2Seq,
        )
        from datasets import Dataset
    except ImportError:
        print("❌ Install transformers + datasets")
        return

    if not config.TRANSLITERATION_PAIRS.exists():
        print("❌ Transliteration pairs not found. Run a11_transliteration_pairs.py first.")
        return

    pairs = [
        json.loads(l) for l in config.TRANSLITERATION_PAIRS.read_text(encoding="utf-8").splitlines() if l
    ]
    tokenizer = AutoTokenizer.from_pretrained(config.MT5_MODEL)

    def preprocess(batch):
        inputs = tokenizer(
            [f"transliterate: {r}" for r in batch["roman"]],
            max_length=64, truncation=True, padding="max_length",
        )
        labels = tokenizer(
            batch["hindi"],
            max_length=64, truncation=True, padding="max_length",
        )
        inputs["labels"] = labels["input_ids"]
        return inputs

    ds = Dataset.from_list(pairs).map(preprocess, batched=True)
    split = ds.train_test_split(test_size=0.1, seed=42)
    model = AutoModelForSeq2SeqLM.from_pretrained(config.MT5_MODEL)

    args = Seq2SeqTrainingArguments(
        output_dir=str(MODEL_OUTPUT_DIR),
        num_train_epochs=5,
        per_device_train_batch_size=16,
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
    print(f"✅ Transliterator model saved → {MODEL_OUTPUT_DIR}")


# ── Streamlit UI ───────────────────────────────────────────────────────────────
def _streamlit_app():
    import streamlit as st

    st.set_page_config(page_title="हिन्दी Transliteration Tool", page_icon="🔤")
    st.title("🔤 Roman → हिन्दी Transliteration Tool")

    mode = st.radio("Mode:", ["Single Word/Phrase", "Batch Convert"])

    if mode == "Single Word/Phrase":
        inp = st.text_input("Roman/English text:", value="computer")
        if st.button("Transliterate", type="primary"):
            with st.spinner("Converting…"):
                result = transliterate(inp)
            st.markdown(f"### Result: `{result}`")
            # Feedback
            correction = st.text_input("Correct transliteration (if wrong):")
            if correction and st.button("Submit Correction"):
                fb = config.DATA_DIR / "transliteration_feedback.jsonl"
                with open(fb, "a", encoding="utf-8") as f:
                    f.write(json.dumps(
                        {"roman": inp, "predicted": result, "correction": correction},
                        ensure_ascii=False,
                    ) + "\n")
                st.success("Correction saved! धन्यवाद।")
    else:
        batch_text = st.text_area("एक पंक्ति में एक Roman शब्द डालें:")
        if st.button("Batch Convert", type="primary"):
            words = [w.strip() for w in batch_text.splitlines() if w.strip()]
            if words:
                with st.spinner(f"{len(words)} words convert हो रहे हैं…"):
                    results = [(w, transliterate(w)) for w in words]
                import pandas as pd
                st.dataframe(pd.DataFrame(results, columns=["Roman", "हिन्दी"]))
            else:
                st.warning("कृपया कम से कम एक शब्द डालें।")


@click.command()
@click.option("--train", "do_train", is_flag=True)
@click.option("--convert", default=None)
def cli(do_train: bool, convert: str | None):
    if do_train:
        train_mt5()
    if convert:
        print(f"{convert} → {transliterate(convert)}")


if __name__ == "__main__":
    import sys as _sys
    if "streamlit" in _sys.modules or "STREAMLIT_SERVER_PORT" in __import__("os").environ:
        _streamlit_app()
    else:
        cli()
