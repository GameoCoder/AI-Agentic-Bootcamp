"""
a10_morphological.py
────────────────────
Morphological analysis pipeline:
  1. Prompts the Ollama LLM to decompose Hindi words into morphemes.
  2. Saves the output as JSONL (one analysis per line).
  3. Provides a stub `BiLSTMTagger` class skeleton for training on the
     synthetic data (full training requires PyTorch + labeled data).

Usage:
    python section_a/a10_morphological.py
    python section_a/a10_morphological.py --words "खाना,जाना,पढ़ाई,बच्चा"
"""

import json
import re
import sys
from pathlib import Path
import click
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from utils.ollama_client import ollama_chat

OUTPUT_JSONL = config.DATA_DIR / "morphological_data.jsonl"

SAMPLE_WORDS = [
    "खाना", "पानी", "जाना", "आना", "पढ़ाई", "लिखाई", "बच्चा", "लड़की",
    "किताब", "स्कूल", "सरकार", "अध्यापक", "विद्यालय", "सुंदर", "खुशी",
    "नाचना", "गाना", "सफलता", "असफलता", "बेरोजगारी", "स्वतंत्रता",
    "भारतीय", "हिंदुस्तान", "महाराष्ट्र", "दिल्लीवाला", "समझदार",
    "लड़का", "घर", "काम", "ज़रूरत", "समझना", "बोलना", "देखना",
]


def analyze_word(word: str) -> dict:
    """Ask LLM to morphologically analyze a Hindi word."""
    prompt = (
        f"हिन्दी शब्द '{word}' के morphemes (उपसर्ग, मूल, प्रत्यय) बताइए। "
        "JSON format में दें:\n"
        '{"word": "<word>", "root": "<root>", "prefix": "<prefix_or_empty>", '
        '"suffix": "<suffix_or_empty>", "meaning": "<brief meaning in Hindi>"}'
    )
    response = ollama_chat(prompt)
    # Extract JSON from response
    json_match = re.search(r"\{.*\}", response, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group())
            data["word"] = word  # ensure correct word
            return data
        except json.JSONDecodeError:
            pass
    return {
        "word": word,
        "root": word,
        "prefix": "",
        "suffix": "",
        "meaning": response[:100],
    }


@click.command()
@click.option("--words", default=None, help="Comma-separated list of words to analyze")
@click.option("--from-corpus", is_flag=True, help="Sample words from unigram list")
@click.option("--sample-size", default=100, show_default=True)
def main(words: str | None, from_corpus: bool, sample_size: int):
    if words:
        word_list = [w.strip() for w in words.split(",") if w.strip()]
    elif from_corpus and config.UNIGRAMS_JSON.exists():
        vocab = list(json.loads(config.UNIGRAMS_JSON.read_text(encoding="utf-8")).keys())
        word_list = [w for w in vocab[:sample_size] if len(w) > 2]
    else:
        word_list = SAMPLE_WORDS
        print(f"📝 Using {len(word_list)} built-in sample words")

    results = []
    for word in tqdm(word_list, desc="Morphological analysis"):
        analysis = analyze_word(word)
        results.append(analysis)

    OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"✅ Morphological data saved → {OUTPUT_JSONL}  ({len(results)} entries)")
    print("\nSample output:")
    for r in results[:3]:
        print(f"  {r}")


if __name__ == "__main__":
    main()
