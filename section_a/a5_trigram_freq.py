"""
a5_trigram_freq.py
──────────────────
Counts three-word sequence (trigram) frequencies from the tokenized
corpus. LLM validates a sample for linguistic plausibility.

Usage:
    python section_a/a5_trigram_freq.py
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


def build_trigram_freq(tokenized_path: Path) -> Counter:
    counter: Counter = Counter()
    for line in tqdm(
        tokenized_path.read_text(encoding="utf-8").splitlines(),
        desc="Counting trigrams",
    ):
        tokens = line.strip().split()
        for i in range(len(tokens) - 2):
            counter[(tokens[i], tokens[i + 1], tokens[i + 2])] += 1
    return counter


def llm_validate_trigrams(trigrams: list[tuple], sample: int = 30) -> list[tuple]:
    """Ask LLM which trigrams are plausible Hindi phrases."""
    sample_list = trigrams[:sample]
    numbered = "\n".join(f"{i+1}. {' '.join(t)}" for i, t in enumerate(sample_list))
    prompt = (
        "निम्नलिखित हिन्दी trigrams में से भाषाई रूप से उचित trigrams के "
        "नंबर दें (comma-separated)। यदि सभी ठीक हैं तो 'all' लिखें।\n\n"
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

    freq = build_trigram_freq(config.TOKENIZED_CORPUS)
    sorted_freq = {
        f"{a} {b} {c}": cnt
        for (a, b, c), cnt in sorted(freq.items(), key=lambda x: -x[1])
    }
    print(f"   {len(sorted_freq):,} unique trigrams found")

    top_trigram_tuples = [tuple(k.split()) for k in list(sorted_freq.keys())[:60]]
    print("🤖 LLM validating top 30 trigrams…")
    valid = llm_validate_trigrams(top_trigram_tuples)   # type: ignore[arg-type]
    print(f"   {len(valid)}/{min(30, len(top_trigram_tuples))} trigrams passed validation")

    config.TRIGRAMS_JSON.write_text(
        json.dumps(sorted_freq, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"✅ Trigram frequencies saved → {config.TRIGRAMS_JSON}")


if __name__ == "__main__":
    main()
