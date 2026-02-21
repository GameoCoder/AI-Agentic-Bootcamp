"""
c4_spell_checker.py
───────────────────
Hindi Spell Checker & Corrector.
  - Loads unigram dictionary from data/unigrams.json.
  - Detects out-of-vocabulary (OOV) words.
  - Queries Ollama LLM for context-aware correction suggestions.

Usage:
    python section_c/c4_spell_checker.py --text "मेरा नाम रमेश हे"
    python section_c/c4_spell_checker.py          # interactive
"""

import json
import re
import sys
from pathlib import Path
import click

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from utils.ollama_client import ollama_chat

try:
    from indicnlp.tokenize import indic_tokenize
    def tokenize(s): return indic_tokenize.trivial_tokenize(s, lang="hi")
except ImportError:
    def tokenize(s): return s.split()

HINDI_RE = re.compile(r"[\u0900-\u097F]")


def load_dictionary() -> set[str]:
    if config.UNIGRAMS_JSON.exists():
        data = json.loads(config.UNIGRAMS_JSON.read_text(encoding="utf-8"))
        # Include only words with freq > 1 (noise reduction)
        return {w for w, c in data.items() if c > 1}
    return set()


def is_hindi_word(word: str) -> bool:
    return any(HINDI_RE.match(c) for c in word)


def suggest_corrections(sentence: str, oov_words: list[str]) -> str:
    """Ask LLM to suggest corrections for OOV words within context."""
    oov_str = ", ".join(f"'{w}'" for w in oov_words)
    prompt = (
        f"इस हिन्दी वाक्य में संभवतः गलत शब्द हैं: {oov_str}\n"
        f"पूरा वाक्य: '{sentence}'\n\n"
        f"सही हिन्दी वाक्य लिखें (केवल सही वाक्य, कोई विवरण नहीं):"
    )
    return ollama_chat(prompt)


def spell_check(sentence: str, dictionary: set[str]) -> dict:
    tokens = tokenize(sentence)
    oov = [t for t in tokens if is_hindi_word(t) and t not in dictionary]

    if not oov:
        return {
            "input": sentence,
            "oov_words": [],
            "corrected": sentence,
            "has_errors": False,
        }

    corrected = suggest_corrections(sentence, oov)
    return {
        "input": sentence,
        "oov_words": oov,
        "corrected": corrected,
        "has_errors": True,
    }


@click.command()
@click.option("--text", default=None, help="Sentence to check")
def main(text: str | None):
    dictionary = load_dictionary()
    if dictionary:
        print(f"📖 Dictionary loaded: {len(dictionary):,} words")
    else:
        print("⚠  Dictionary empty – run a3_unigram_freq.py first. Using LLM-only mode.")

    if text:
        result = spell_check(text, dictionary)
        print(f"\nInput:     {result['input']}")
        if result["has_errors"]:
            print(f"OOV words: {result['oov_words']}")
            print(f"Corrected: {result['corrected']}")
        else:
            print("✅ No spelling errors detected.")
    else:
        print("\n🔡 Hindi Spell Checker (type 'quit' to exit)\n")
        while True:
            line = input("वाक्य: ").strip()
            if line.lower() in ("quit", "exit"):
                break
            result = spell_check(line, dictionary)
            if result["has_errors"]:
                print(f"  OOV: {result['oov_words']}")
                print(f"  सुधरा वाक्य: {result['corrected']}")
            else:
                print("  ✅ सही है।")
            print()


if __name__ == "__main__":
    main()
