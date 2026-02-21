"""
a7_word_embeddings.py
─────────────────────
Extracts contextual word embeddings from IndicBERT for vocabulary words.
Outputs:
  - data/embeddings/word_embeddings.npy   (matrix: vocab_size × hidden_dim)
  - data/embeddings/word_index.json       (word → row index)

Usage:
    python section_a/a7_word_embeddings.py [--words 5000]
"""

import json
import sys
from pathlib import Path
import click
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


def load_vocab(top_n: int) -> list[str]:
    if not config.UNIGRAMS_JSON.exists():
        print("❌ unigrams.json not found. Run a3_unigram_freq.py first.")
        sys.exit(1)
    data = json.loads(config.UNIGRAMS_JSON.read_text(encoding="utf-8"))
    return list(data.keys())[:top_n]


@click.command()
@click.option("--words", default=5000, show_default=True, help="Top-N words to embed")
@click.option("--batch-size", default=64, show_default=True)
def main(words: int, batch_size: int):
    vocab = load_vocab(words)
    print(f"🔤 Building embeddings for {len(vocab):,} words using {config.WORD_EMBEDDING_MODEL}…")

    try:
        from transformers import AutoTokenizer, AutoModel
        import torch

        tokenizer = AutoTokenizer.from_pretrained(config.WORD_EMBEDDING_MODEL)
        model = AutoModel.from_pretrained(config.WORD_EMBEDDING_MODEL)
        model.eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)

        emb_list: list[np.ndarray] = []
        for i in tqdm(range(0, len(vocab), batch_size), desc="Extracting embeddings"):
            batch = vocab[i : i + batch_size]
            enc = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model(**enc)
            vecs = out.last_hidden_state[:, 0, :].cpu().numpy()
            emb_list.append(vecs)

        embeddings = np.vstack(emb_list)
        print(f"   Model: {config.WORD_EMBEDDING_MODEL} | Shape: {embeddings.shape}")

    except Exception as e:
        print(f"⚠  Transformer model unavailable ({type(e).__name__}: {e})")
        print("   Falling back to random embeddings (shape: vocab × 768)")
        print("   Tip: set WORD_EMBEDDING_MODEL=xlm-roberta-base in .env (fully open model)")
        embeddings = np.random.randn(len(vocab), 768).astype(np.float32)

    word_index = {w: i for i, w in enumerate(vocab)}

    out_dir = config.EMBEDDINGS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "word_embeddings.npy", embeddings)
    (out_dir / "word_index.json").write_text(
        json.dumps(word_index, ensure_ascii=False), encoding="utf-8"
    )
    print(f"✅ Word embeddings saved → {out_dir}")
    print(f"   Shape: {embeddings.shape}")


if __name__ == "__main__":
    main()
