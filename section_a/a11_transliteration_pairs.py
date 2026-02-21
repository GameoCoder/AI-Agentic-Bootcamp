"""
a11_transliteration_pairs.py
────────────────────────────
Generates Roman ↔ Devanagari transliteration pairs by:
  1. Prompting the Ollama LLM to transliterate common loanwords.
  2. Crawling Wikipedia interlingual links to extract title-level pairs.
  3. Saving results as JSONL: { "roman": "...", "hindi": "..." }

Usage:
    python section_a/a11_transliteration_pairs.py
    python section_a/a11_transliteration_pairs.py --wiki --llm-count 200
"""

import json
import re
import sys
from pathlib import Path
import click
import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from utils.ollama_client import ollama_chat

# Common English/Roman loanwords used in Hindi everyday speech
LOANWORDS = [
    "computer", "mobile", "internet", "school", "doctor", "hospital",
    "train", "bus", "station", "market", "office", "manager", "cricket",
    "football", "cinema", "ticket", "hotel", "restaurant", "airport",
    "university", "college", "professor", "engineer", "police", "minister",
    "government", "election", "newspaper", "television", "radio", "camera",
    "battery", "charger", "password", "email", "website", "download",
    "software", "hardware", "keyboard", "mouse", "printer", "screen",
]


def llm_transliterate_batch(words: list[str]) -> list[dict]:
    """Ask LLM to transliterate each word into Hindi Devanagari script."""
    pairs: list[dict] = []
    for i in range(0, len(words), 10):
        batch = words[i : i + 10]
        prompt = (
            "निम्नलिखित अंग्रेजी/Roman शब्दों को हिन्दी (Devanagari) script में लिखें। "
            "प्रत्येक शब्द के लिए एक नई लाइन पर 'roman: hindi' format में लिखें।\n\n"
            + "\n".join(batch)
        )
        response = ollama_chat(prompt)
        for line in response.splitlines():
            if ":" in line:
                parts = line.split(":", 1)
                roman = re.sub(r"^\d+\.\s*", "", parts[0]).strip().lower()
                hindi = parts[1].strip()
                if roman and hindi and re.search(r"[\u0900-\u097F]", hindi):
                    pairs.append({"roman": roman, "hindi": hindi})
    return pairs


def fetch_wikipedia_pairs(limit: int = 300) -> list[dict]:
    """
    Query Hindi Wikipedia API for article titles and fetch their
    English counterparts via langlinks to get Roman ↔ Devanagari pairs.
    """
    pairs: list[dict] = []
    api = "https://hi.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "random",
        "rnnamespace": 0,
        "rnlimit": min(limit, 500),
        "prop": "langlinks",
        "lllang": "en",
        "format": "json",
    }
    try:
        resp = requests.get(api, params=params, timeout=15)
        resp.raise_for_status()
        pages = resp.json().get("query", {}).get("random", [])
        titles_hi = [p["title"] for p in pages]

        # Now get EN langlinks for these titles
        for title in tqdm(titles_hi[:limit], desc="Fetching Wikipedia pairs"):
            p2 = {
                "action": "query",
                "titles": title,
                "prop": "langlinks",
                "lllang": "en",
                "format": "json",
            }
            try:
                r2 = requests.get(api, params=p2, timeout=10)
                data = r2.json().get("query", {}).get("pages", {})
                for page in data.values():
                    ll = page.get("langlinks", [])
                    if ll:
                        en_title = ll[0]["*"]
                        pairs.append({"roman": en_title, "hindi": title})
            except Exception:
                continue
    except Exception as e:
        print(f"⚠  Wikipedia fetch failed: {e}")

    return pairs


@click.command()
@click.option("--llm-count", default=len(LOANWORDS), show_default=True,
              help="Number of loanwords for LLM transliteration")
@click.option("--wiki", is_flag=True, default=False, help="Also crawl Wikipedia for pairs")
@click.option("--wiki-limit", default=200, show_default=True)
def main(llm_count: int, wiki: bool, wiki_limit: int):
    all_pairs: list[dict] = []

    print(f"🤖 LLM transliterating {llm_count} loanwords…")
    llm_pairs = llm_transliterate_batch(LOANWORDS[:llm_count])
    all_pairs.extend(llm_pairs)
    print(f"   Got {len(llm_pairs)} LLM pairs")

    if wiki:
        print(f"🌐 Fetching {wiki_limit} Wikipedia interlingual pairs…")
        wiki_pairs = fetch_wikipedia_pairs(wiki_limit)
        all_pairs.extend(wiki_pairs)
        print(f"   Got {len(wiki_pairs)} Wikipedia pairs")

    # Deduplicate
    seen = set()
    unique: list[dict] = []
    for p in all_pairs:
        key = p["roman"].lower()
        if key not in seen:
            seen.add(key)
            unique.append(p)

    config.TRANSLITERATION_PAIRS.parent.mkdir(parents=True, exist_ok=True)
    with open(config.TRANSLITERATION_PAIRS, "w", encoding="utf-8") as f:
        for p in unique:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"✅ {len(unique)} transliteration pairs saved → {config.TRANSLITERATION_PAIRS}")


if __name__ == "__main__":
    main()
