"""
a8_sentence_embeddings.py
─────────────────────────
Encodes all corpus sentences using LaBSE (sentence-transformers).
Saves:
  - data/faiss_index/   (FAISS index for similarity search)
  - data/embeddings/sentences.npy (raw embedding matrix)
  - data/embeddings/sentence_lines.txt (one sentence per line, aligned with matrix)

Usage:
    python section_a/a8_sentence_embeddings.py [--batch-size 128]
"""

import sys
from pathlib import Path
import click
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


@click.command()
@click.option("--batch-size", default=128, show_default=True)
@click.option("--max-sentences", default=0, help="0 = all", show_default=True)
def main(batch_size: int, max_sentences: int):
    from sentence_transformers import SentenceTransformer
    import faiss

    if not config.RAW_CORPUS.exists():
        print("❌ raw_corpus.txt not found. Run a1_corpus_crawler.py first.")
        return

    sentences = config.RAW_CORPUS.read_text(encoding="utf-8").splitlines()
    if max_sentences > 0:
        sentences = sentences[:max_sentences]
    print(f"📦 Encoding {len(sentences):,} sentences with {config.SENTENCE_TRANSFORMER_MODEL}…")

    model = SentenceTransformer(config.SENTENCE_TRANSFORMER_MODEL)
    embeddings = model.encode(
        sentences,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    out_dir = config.EMBEDDINGS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "sentences.npy", embeddings)
    (out_dir / "sentence_lines.txt").write_text(
        "\n".join(sentences), encoding="utf-8"
    )
    print(f"✅ Sentence embeddings saved → {out_dir / 'sentences.npy'}  {embeddings.shape}")

    # Build FAISS index
    print("🗂  Building FAISS index…")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)          # Inner-Product (cosine since normalised)
    index.add(embeddings.astype(np.float32))
    faiss_dir = config.FAISS_INDEX
    faiss_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(faiss_dir / "index.faiss"))
    print(f"✅ FAISS index saved → {faiss_dir / 'index.faiss'}  ({index.ntotal} vectors)")


if __name__ == "__main__":
    main()
