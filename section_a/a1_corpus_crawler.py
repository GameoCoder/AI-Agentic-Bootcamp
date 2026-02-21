"""
a1_corpus_crawler.py
────────────────────
Extracts Hindi sentences from the local hiwiki Wikipedia XML bz2 dump.
Uses mwparserfromhell to strip wiki markup, then calls the Ollama
(qwen2:0.5b) LLM to clean / filter non-Hindi lines and fix encoding.
Also generates a configurable number of synthetic Hindi sentences.

Usage:
    python section_a/a1_corpus_crawler.py
    python section_a/a1_corpus_crawler.py --max-articles 1000 --synthetic 200
"""

import bz2
import json
import re
import sys
import click
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from utils.ollama_client import ollama_chat

try:
    import mwparserfromhell
    HAS_MWPARSER = True
except ImportError:
    HAS_MWPARSER = False

try:
    from xml.etree import cElementTree as ET
except ImportError:
    from xml.etree import ElementTree as ET

from tqdm import tqdm

HINDI_RE = re.compile(r"[\u0900-\u097F]")


def is_predominantly_hindi(text: str, threshold: float = 0.3) -> bool:
    """Return True if ≥ threshold fraction of chars are Devanagari."""
    if not text.strip():
        return False
    hindi_chars = sum(1 for c in text if HINDI_RE.match(c))
    return (hindi_chars / len(text)) >= threshold


def clean_wiki_markup(raw: str) -> str:
    """Strip MediaWiki markup using mwparserfromhell or regex fallback."""
    if HAS_MWPARSER:
        wikicode = mwparserfromhell.parse(raw)
        return wikicode.strip_code()
    # Regex fallback: remove {{...}}, [[...]], <...>
    text = re.sub(r"\{\{.*?\}\}", "", raw, flags=re.DOTALL)
    text = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]*)\]\]", r"\1", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"={2,}.*?={2,}", "", text)
    return text


def extract_sentences_from_xml(xml_path: Path, max_articles: int) -> list[str]:
    """Stream-parse the bz2 XML and yield clean Hindi sentences."""
    sentences: list[str] = []
    ns_wiki = "http://www.mediawiki.org/xml/DTD/mediawiki"

    article_count = 0
    with bz2.open(xml_path, "rb") as f:
        context = ET.iterparse(f, events=("end",))
        for event, elem in tqdm(context, desc="Parsing Wikipedia XML"):
            tag = elem.tag.split("}")[-1]  # strip namespace
            if tag == "text" and elem.text:
                raw = elem.text
                cleaned = clean_wiki_markup(raw)

                for line in cleaned.splitlines():
                    line = line.strip()
                    if len(line) > 20 and is_predominantly_hindi(line):
                        sentences.append(line)

                article_count += 1
                elem.clear()

                if article_count >= max_articles:
                    break

    return sentences


def llm_filter_sentences(sentences: list[str], batch_size: int = 1) -> list[str]:
    """
    Ask the LLM to validate batches of sentences.
    Returns the subset that the LLM confirms are valid Hindi.
    """
    valid: list[str] = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i : i + batch_size]
        numbered = "\n".join(f"{j+1}. {s}" for j, s in enumerate(batch))
        prompt = (
            "नीचे दिए गए वाक्यों में से केवल वे वाक्य बताइए जो सही हिन्दी हैं। "
            "उनके नंबर comma-separated लिखें (जैसे: 1,3,5)। "
            "यदि सभी सही हैं तो 'all' लिखें।\n\n"
            f"{numbered}"
        )
        response = ollama_chat(prompt)
        text = response.strip().lower()
        if "all" in text:
            valid.extend(batch)
        else:
            nums = re.findall(r"\d+", text)
            for n in nums:
                idx = int(n) - 1
                if 0 <= idx < len(batch):
                    valid.append(batch[idx])
    return valid


def generate_synthetic_sentences(count: int, topics: list[str] | None = None) -> list[str]:
    """Use the LLM to generate diverse synthetic Hindi sentences."""
    if topics is None:
        topics = [
            "खेल", "राजनीति", "विज्ञान", "प्रकृति", "शिक्षा",
            "स्वास्थ्य", "मनोरंजन", "अर्थव्यवस्था", "संस्कृति", "प्रौद्योगिकी"
        ]
    sentences: list[str] = []
    per_topic = max(1, count // len(topics))

    for topic in tqdm(topics, desc="Generating synthetic sentences"):
        prompt = (
            f"हिन्दी में '{topic}' विषय पर {per_topic} अलग-अलग, "
            f"सरल और स्पष्ट वाक्य लिखें। प्रत्येक वाक्य एक नई पंक्ति पर हो।"
        )
        response = ollama_chat(prompt)
        for line in response.splitlines():
            line = line.strip()
            # Remove leading numbering
            line = re.sub(r"^\d+[\.\)]\s*", "", line)
            if len(line) > 10 and is_predominantly_hindi(line):
                sentences.append(line)
        if len(sentences) >= count:
            break

    return sentences[:count]


@click.command()
@click.option("--max-articles", default=5000, show_default=True, help="Max wiki articles to parse")
@click.option("--synthetic", default=config.SYNTHETIC_SENT_COUNT, show_default=True, help="Synthetic sentences to generate")
@click.option("--output", default=str(config.RAW_CORPUS), show_default=True)
@click.option("--skip-llm-filter", is_flag=True, default=False, help="Skip Ollama filtering step")
def main(max_articles: int, synthetic: int, output: str, skip_llm_filter: bool):
    click.echo(f"📂 Reading Wikipedia XML: {config.WIKI_XML_PATH}")
    sentences = extract_sentences_from_xml(config.WIKI_XML_PATH, max_articles)
    click.echo(f"   Extracted {len(sentences):,} candidate sentences")

    if not skip_llm_filter:
        click.echo("🤖 LLM filtering sentences…")
        sentences = llm_filter_sentences(sentences)
        click.echo(f"   {len(sentences):,} sentences passed LLM filter")

    if synthetic > 0:
        click.echo(f"✨ Generating {synthetic} synthetic Hindi sentences…")
        synth = generate_synthetic_sentences(synthetic)
        sentences.extend(synth)
        click.echo(f"   Total after augmentation: {len(sentences):,}")

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(sentences))

    click.echo(f"✅ Corpus saved → {out_path}  ({len(sentences):,} sentences)")


if __name__ == "__main__":
    main()
