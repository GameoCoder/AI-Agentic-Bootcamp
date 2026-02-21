"""
a2_tokenizer.py
───────────────
Tokenizes the raw Hindi corpus using the Indic NLP Library.
Falls back to whitespace tokenization if the library isn't available.

Usage:
    python section_a/a2_tokenizer.py
"""

import sys
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

try:
    from indicnlp.tokenize import indic_tokenize
    HAS_INDIC = True
except ImportError:
    HAS_INDIC = False
    print("⚠  indic-nlp-library not found – falling back to whitespace tokenisation.")


def tokenize_sentence(sentence: str) -> list[str]:
    """Return a list of tokens for a Hindi sentence."""
    if HAS_INDIC:
        return indic_tokenize.trivial_tokenize(sentence, lang="hi")
    return sentence.split()


def tokenize_corpus(input_path: Path, output_path: Path) -> None:
    """Read raw sentences, tokenize each, write space-joined tokens per line."""
    if not input_path.exists():
        print(f"❌ Raw corpus not found: {input_path}")
        print("   Run section_a/a1_corpus_crawler.py first.")
        return

    lines = input_path.read_text(encoding="utf-8").splitlines()
    print(f"📖 Tokenizing {len(lines):,} sentences…")

    tokenized = []
    for line in tqdm(lines, desc="Tokenizing"):
        tokens = tokenize_sentence(line.strip())
        if tokens:
            tokenized.append(" ".join(tokens))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(tokenized), encoding="utf-8")
    print(f"✅ Tokenized corpus saved → {output_path}  ({len(tokenized):,} lines)")


if __name__ == "__main__":
    tokenize_corpus(config.RAW_CORPUS, config.TOKENIZED_CORPUS)
