"""
a9_ngram_lm.py
──────────────
LLM-based language model wrapper.
Uses the Ollama qwen2:0.5b LLM to:
  1. Estimate log-probability / perplexity of a sentence.
  2. Generate text continuations given a prefix.

Since qwen2 doesn't expose raw logprobs, perplexity is approximated via
a prompt that asks the model to rate sentence fluency on a 1-10 scale,
then converts to a pseudo-perplexity (lower = better).

Usage:
    python section_a/a9_ngram_lm.py --text "भारत एक महान देश है"
    python section_a/a9_ngram_lm.py --generate --prefix "आज का मौसम"
"""

import sys
import re
import math
import click
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from utils.ollama_client import ollama_chat


def estimate_perplexity(sentence: str) -> float:
    """
    Ask the LLM to rate sentence fluency 1–10; convert to pseudo-perplexity.
    Fluency 10 → pseudo-perplexity ~1 (perfect)
    Fluency 1  → pseudo-perplexity ~100 (terrible)
    """
    prompt = (
        f"निम्नलिखित हिन्दी वाक्य की भाषाई प्रवाहता का मूल्यांकन 1 से 10 के बीच करें "
        f"(10 = बिल्कुल सही, 1 = बिल्कुल गलत)। केवल एक अंक दें।\n\n"
        f"वाक्य: {sentence}"
    )
    response = ollama_chat(prompt).strip()
    nums = re.findall(r"\d+", response)
    score = int(nums[0]) if nums else 5
    score = max(1, min(10, score))
    # Map: fluency_score → pseudo_perplexity = e^((10 - score))
    return math.exp(10 - score)


def generate_text(prefix: str, max_words: int = 30) -> str:
    """Generate a Hindi sentence continuation from a prefix."""
    prompt = (
        f"निम्नलिखित हिन्दी वाक्यांश को पूरा करें (अधिकतम {max_words} शब्द):\n{prefix}"
    )
    return ollama_chat(prompt)


def evaluate_corpus_perplexity(held_out_path: Path, sample: int = 100) -> float:
    """Compute average pseudo-perplexity over held-out sentences."""
    lines = held_out_path.read_text(encoding="utf-8").splitlines()[:sample]
    total_pp = sum(estimate_perplexity(s) for s in lines if s.strip())
    return total_pp / len(lines)


@click.command()
@click.option("--text", default=None, help="Score a single sentence")
@click.option("--generate", is_flag=True, help="Generate text continuation")
@click.option("--prefix", default="आज का", show_default=True)
@click.option("--eval-corpus", is_flag=True, help="Evaluate average perplexity on tokenized corpus")
@click.option("--sample", default=50, show_default=True)
def main(text: str | None, generate: bool, prefix: str, eval_corpus: bool, sample: int):
    if text:
        pp = estimate_perplexity(text)
        click.echo(f"📊 Pseudo-perplexity of '{text}': {pp:.2f}")

    if generate:
        result = generate_text(prefix)
        click.echo(f"✍  Generated: {prefix} {result}")

    if eval_corpus:
        if not config.TOKENIZED_CORPUS.exists():
            click.echo("❌ Tokenized corpus not found.")
            return
        avg_pp = evaluate_corpus_perplexity(config.TOKENIZED_CORPUS, sample=sample)
        click.echo(f"📉 Average pseudo-perplexity over {sample} held-out sentences: {avg_pp:.2f}")


if __name__ == "__main__":
    main()
