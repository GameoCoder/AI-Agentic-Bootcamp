"""
a4_bigram_freq.py
─────────────────
Counts adjacent word-pair (bigram) frequencies from the tokenized
corpus and saves sorted JSON. LLM filters a sample of top bigrams
for tokenization artefacts.

Usage:
    python section_a/a4_bigram_freq.py
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from utils.ollama_client import ollama_chat


def build_bigram_freq(tokenized_path: Path) -> Counter:
    counter: Counter = Counter()
    for line in tqdm(
        tokenized_path.read_text(encoding="utf-8").splitlines(),
        desc="Counting bigrams",
    ):
        tokens = line.strip().split()
        for i in range(len(tokens) - 1):
            counter[(tokens[i], tokens[i + 1])] += 1
    return counter


def llm_filter_bigrams(bigrams: list[tuple[str, str]], sample: int = 40) -> list[tuple[str, str]]:
    """Ask LLM which bigrams are linguistically meaningful."""
    sample_list = bigrams[:sample]
    numbered = "\n".join(f"{i+1}. {a} {b}" for i, (a, b) in enumerate(sample_list))
    prompt = (
        "निम्नलिखित हिन्दी bigrams में से केवल भाषाई रूप से अर्थपूर्ण bigrams के "
        "नंबर बताएं (comma-separated)। यदि सभी सही हैं तो 'all' लिखें।\n\n"
        f"{numbered}"
    )
    response = ollama_chat(prompt).strip().lower()
    if "all" in response:
        return sample_list
    nums = re.findall(r"\d+", response)
    return [sample_list[int(n) - 1] for n in nums if 0 < int(n) <= len(sample_list)]


def main():
    if not config.TOKENIZED_CORPUS.exists():
        print("❌ Tokenized corpus not found. Run a2_tokenizer.py first.")
        return

    freq = build_bigram_freq(config.TOKENIZED_CORPUS)
    # Convert tuple keys to "w1 w2" strings for JSON serialisation
    sorted_freq = {
        f"{a} {b}": c
        for (a, b), c in sorted(freq.items(), key=lambda x: -x[1])
    }
    print(f"   {len(sorted_freq):,} unique bigrams found")

    top_bigram_tuples = [tuple(k.split(None, 1)) for k in list(sorted_freq.keys())[:80]]
    print("🤖 LLM filtering top 40 bigrams…")
    valid = llm_filter_bigrams(top_bigram_tuples)   # type: ignore[arg-type]
    print(f"   {len(valid)}/{min(40, len(top_bigram_tuples))} bigrams passed filter")

    config.BIGRAMS_JSON.write_text(
        json.dumps(sorted_freq, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"✅ Bigram frequencies saved → {config.BIGRAMS_JSON}")


if __name__ == "__main__":
    main()
