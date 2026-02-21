"""
tests/test_section_d.py
────────────────────────
Unit tests for Section D analysis tools.
"""

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestD1ReadabilityAnalyzer(unittest.TestCase):
    def test_split_sentences(self):
        from section_d.d1_readability_analyzer import split_sentences
        text = "भारत एक महान देश है। यहाँ अनेक भाषाएँ हैं।"
        sentences = split_sentences(text)
        self.assertEqual(len(sentences), 2)

    def test_compute_metrics_basic(self):
        from section_d.d1_readability_analyzer import compute_metrics
        text = "राम ने स्कूल में पढ़ाई की। वह बहुत मेहनती था।"
        vocab = {"राम", "ने", "स्कूल", "में", "पढ़ाई", "की", "वह", "बहुत", "मेहनती", "था"}
        metrics = compute_metrics(text, vocab)
        self.assertIn("avg_sentence_length", metrics)
        self.assertIn("oov_rate", metrics)
        self.assertGreater(metrics["total_words"], 0)
        self.assertEqual(metrics["total_sentences"], 2)

    def test_oov_rate_zero_for_full_vocab(self):
        from section_d.d1_readability_analyzer import compute_metrics
        text = "राम गया।"
        vocab = {"राम", "गया"}
        metrics = compute_metrics(text, vocab)
        self.assertEqual(metrics["oov_rate"], 0.0)

    def test_fk_score_within_range(self):
        from section_d.d1_readability_analyzer import compute_metrics
        text = " ".join(["भारत"] * 20 + [" है।"])
        metrics = compute_metrics(text, {"भारत", "है"})
        self.assertGreaterEqual(metrics["fk_reading_ease"], 0)
        self.assertLessEqual(metrics["fk_reading_ease"], 100)

    @patch("section_d.d1_readability_analyzer.ollama_chat",
           return_value="5|मध्यम कठिनाई वाला पाठ")
    def test_llm_difficulty_score_format(self, _):
        from section_d.d1_readability_analyzer import llm_difficulty_score
        result = llm_difficulty_score("भारत में अनेक भाषाएँ हैं।")
        self.assertIn("llm_score", result)
        self.assertIn("llm_reason", result)
        self.assertGreaterEqual(result["llm_score"], 1)
        self.assertLessEqual(result["llm_score"], 10)


class TestD2CorpsStats(unittest.TestCase):
    def test_vocab_growth_returns_empty_on_missing(self):
        from section_d.d2_corpus_stats_dashboard import vocab_growth_data
        import config
        # Temporarily ensure file doesn't exist
        if not config.TOKENIZED_CORPUS.exists():
            df = vocab_growth_data()
            self.assertTrue(df.empty)


if __name__ == "__main__":
    unittest.main()
