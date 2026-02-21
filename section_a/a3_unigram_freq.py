"""
a3_unigram_freq.py
──────────────────
Counts word (unigram) frequencies from the tokenized corpus and
saves a sorted JSON mapping { word: count }.

Usage:
    python section_a/a3_unigram_freq.py
"""

import json
import sys
from collections import Counter
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from utils.ollama_client import ollama_chat


def build_unigram_freq(tokenized_path: Path) -> Counter:
    """Return a Counter of all tokens in the tokenized corpus."""
    counter: Counter = Counter()
    lines = tokenized_path.read_text(encoding="utf-8").splitlines()
    for line in tqdm(lines, desc="Counting unigrams"):
        counter.update(line.strip().split())
    return counter


def llm_suggest_stopwords(top_words: list[str]) -> list[str]:
    """Ask the LLM which of the top-frequency words are function words."""
    words_str = ", ".join(top_words[:60])
    prompt = (
        f"इन हिन्दी शब्दों में से stop words (function words जैसे सर्वनाम, "
        f"परसर्ग, संयोजन) बताइए:\n{words_str}\n"
        "केवल stop words की comma-separated सूची दें।"
    )
    response = ollama_chat(prompt)
    return [w.strip() for w in response.split(",") if w.strip()]


def main():
    if not config.TOKENIZED_CORPUS.exists():
        print("❌ Tokenized corpus not found. Run a2_tokenizer.py first.")
        return

    freq = build_unigram_freq(config.TOKENIZED_CORPUS)
    print(f"   Vocabulary size: {len(freq):,} unique tokens")

    # Save sorted by frequency descending
    sorted_freq = dict(sorted(freq.items(), key=lambda x: -x[1]))
    config.UNIGRAMS_JSON.parent.mkdir(parents=True, exist_ok=True)
    config.UNIGRAMS_JSON.write_text(
        json.dumps(sorted_freq, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"✅ Unigram frequencies saved → {config.UNIGRAMS_JSON}")

    # LLM stop word hint
    top_words = list(sorted_freq.keys())[:80]
    print("🤖 Asking LLM for stop word hints from top 80 words…")
    stop_hints = llm_suggest_stopwords(top_words)
    print(f"   LLM suggested {len(stop_hints)} candidate stop words: {stop_hints[:15]}…")


if __name__ == "__main__":
    main()
