"""
a6_stopwords.py
───────────────
Generates a curated Hindi stop word list by:
  1. Asking the Ollama LLM to suggest common Hindi function words.
  2. Cross-referencing with the unigram frequency list (top-N words).
  3. Writing the final merged set to data/stopwords.txt.

Usage:
    python section_a/a6_stopwords.py
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from utils.ollama_client import ollama_chat

# Built-in seed list (always included)
SEED_STOPWORDS = [
    "का", "की", "के", "में", "है", "हैं", "को", "से", "पर", "और",
    "यह", "वह", "ये", "वे", "जो", "कि", "भी", "ने", "तो", "हो",
    "एक", "ही", "था", "थे", "थी", "या", "लेकिन", "अगर", "जब", "तब",
    "कब", "कहाँ", "क्यों", "कैसे", "हम", "आप", "मैं", "तुम", "वो",
    "इस", "उस", "इन", "उन", "जिस", "जिन", "इसे", "उसे", "नहीं",
    "अब", "तक", "बाद", "पहले", "साथ", "लिए", "बहुत", "कोई", "कुछ",
    "सब", "हर", "दोनों", "द्वारा", "रहा", "रही", "रहे", "जा", "आ",
]


def llm_suggest_stopwords() -> list[str]:
    """Ask LLM for common Hindi stop words."""
    prompt = (
        "हिन्दी भाषा के 60 सबसे सामान्य stop words (function words जैसे "
        "सर्वनाम, परसर्ग, संयोजन, क्रिया-सहायिका) की comma-separated सूची दें। "
        "केवल शब्द, कोई विवरण नहीं।"
    )
    response = ollama_chat(prompt)
    return [w.strip() for w in re.split(r"[,\n]", response) if w.strip()]


def frequency_based_stopwords(top_n: int = 100) -> list[str]:
    """Return top-N words from unigram list as stop word candidates."""
    if not config.UNIGRAMS_JSON.exists():
        return []
    data = json.loads(config.UNIGRAMS_JSON.read_text(encoding="utf-8"))
    return list(data.keys())[:top_n]


def main():
    print("🌱 Loading seed stop words…")
    stop_set: set[str] = set(SEED_STOPWORDS)

    print("🤖 Asking LLM for stop word suggestions…")
    llm_words = llm_suggest_stopwords()
    stop_set.update(llm_words)
    print(f"   LLM contributed {len(llm_words)} words")

    print("📊 Cross-referencing with unigram frequency list…")
    freq_candidates = frequency_based_stopwords(top_n=100)
    if freq_candidates:
        # Keep only short words (≤4 chars → likely function words)
        short_freq = [w for w in freq_candidates if len(w) <= 4]
        stop_set.update(short_freq)
        print(f"   Added {len(short_freq)} short high-frequency words")
    else:
        print("   (Unigram list not available yet – skipped)")

    final = sorted(stop_set)
    config.STOPWORDS_TXT.parent.mkdir(parents=True, exist_ok=True)
    config.STOPWORDS_TXT.write_text("\n".join(final), encoding="utf-8")
    print(f"✅ Stop words saved → {config.STOPWORDS_TXT}  ({len(final)} words)")


if __name__ == "__main__":
    main()
