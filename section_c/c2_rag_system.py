"""
c2_rag_system.py
────────────────
Hindi Retrieval-Augmented Generation (RAG) system.
Uses the FAISS index built in a8_sentence_embeddings.py to retrieve
relevant Hindi sentences, then feeds context to the Ollama LLM.

Usage:
    python section_c/c2_rag_system.py                          # interactive
    python section_c/c2_rag_system.py --query "भारत का इतिहास"
"""

import sys
from pathlib import Path
import click

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from utils.ollama_client import ollama_chat


def load_faiss():
    """Load the FAISS index and sentence list."""
    import faiss
    import numpy as np

    index_path = config.FAISS_INDEX / "index.faiss"
    sentences_path = config.EMBEDDINGS_DIR / "sentence_lines.txt"

    if not index_path.exists():
        raise FileNotFoundError(
            f"FAISS index not found: {index_path}\n"
            "Run section_a/a8_sentence_embeddings.py first."
        )

    index = faiss.read_index(str(index_path))
    sentences = sentences_path.read_text(encoding="utf-8").splitlines()
    return index, sentences


def retrieve(query: str, index, sentences: list[str], top_k: int = 5) -> list[str]:
    """Encode query and retrieve top-k similar sentences."""
    from sentence_transformers import SentenceTransformer
    import numpy as np

    model = SentenceTransformer(config.SENTENCE_TRANSFORMER_MODEL)
    q_emb = model.encode([query], normalize_embeddings=True).astype("float32")
    distances, indices = index.search(q_emb, top_k)
    return [sentences[i] for i in indices[0] if 0 <= i < len(sentences)]


def rag_answer(query: str, index, sentences: list[str], top_k: int = 5) -> str:
    """Retrieve context and generate a grounded answer."""
    context_docs = retrieve(query, index, sentences, top_k=top_k)
    context = "\n".join(f"- {d}" for d in context_docs)

    prompt = (
        f"निम्नलिखित संदर्भ का उपयोग करके प्रश्न का उत्तर हिन्दी में दें:\n\n"
        f"संदर्भ:\n{context}\n\n"
        f"प्रश्न: {query}\n\n"
        f"उत्तर:"
    )
    return ollama_chat(prompt)


def answer(query: str) -> str:
    """Public convenience function: answer a question using RAG or LLM fallback."""
    try:
        index, sentences = load_faiss()
        return rag_answer(query, index, sentences, top_k=5)
    except FileNotFoundError:
        return ollama_chat(query)


@click.command()
@click.option("--query", default=None, help="Single query (non-interactive)")
@click.option("--top-k", default=5, show_default=True)
def main(query: str | None, top_k: int):
    try:
        index, sentences = load_faiss()
        print(f"✅ FAISS index loaded ({index.ntotal} vectors)")
    except FileNotFoundError as e:
        print(f"⚠  {e}")
        print("Falling back to LLM-only mode (no retrieval)")
        index, sentences = None, []

    def answer(q: str) -> str:
        if index is not None:
            return rag_answer(q, index, sentences, top_k)
        return ollama_chat(q)

    if query:
        print(f"\nQ: {query}")
        print(f"A: {answer(query)}")
    else:
        print("🔍 Hindi RAG System (type 'quit' to exit)\n")
        while True:
            q = input("प्रश्न: ").strip()
            if q.lower() in ("quit", "exit"):
                break
            print(f"उत्तर: {answer(q)}\n")


if __name__ == "__main__":
    main()
