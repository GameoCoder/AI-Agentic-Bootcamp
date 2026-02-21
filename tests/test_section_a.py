"""
tests/test_section_a.py
────────────────────────
Unit tests for Section A modules.
Uses unittest.mock to avoid actual LLM / filesystem calls.
"""

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestA1CorpusCrawler(unittest.TestCase):
    def test_is_predominantly_hindi_true(self):
        from section_a.a1_corpus_crawler import is_predominantly_hindi
        self.assertTrue(is_predominantly_hindi("भारत एक महान देश है"))

    def test_is_predominantly_hindi_false(self):
        from section_a.a1_corpus_crawler import is_predominantly_hindi
        self.assertFalse(is_predominantly_hindi("This is pure English text"))

    def test_is_predominantly_hindi_mixed(self):
        from section_a.a1_corpus_crawler import is_predominantly_hindi
        # Short English with some Hindi – depends on ratio
        result = is_predominantly_hindi("Hello दुनिया")
        self.assertIsInstance(result, bool)

    def test_clean_wiki_markup_strips_brackets(self):
        from section_a.a1_corpus_crawler import clean_wiki_markup
        raw = "[[दिल्ली]] भारत की [[राजधानी]] है।"
        cleaned = clean_wiki_markup(raw)
        self.assertNotIn("[[", cleaned)
        self.assertNotIn("]]", cleaned)
        self.assertIn("दिल्ली", cleaned)

    @patch("section_a.a1_corpus_crawler.ollama_chat", return_value="all")
    def test_llm_filter_returns_all(self, mock_llm):
        from section_a.a1_corpus_crawler import llm_filter_sentences
        sentences = ["भारत एक देश है।", "हिन्दी हमारी भाषा है।"]
        result = llm_filter_sentences(sentences, batch_size=10)
        self.assertEqual(result, sentences)

    @patch("section_a.a1_corpus_crawler.ollama_chat", return_value="1,2")
    def test_llm_filter_partial(self, mock_llm):
        from section_a.a1_corpus_crawler import llm_filter_sentences
        s = ["वाक्य एक", "वाक्य दो", "sentence three"]
        result = llm_filter_sentences(s, batch_size=10)
        self.assertEqual(len(result), 2)


class TestA2Tokenizer(unittest.TestCase):
    def test_tokenize_whitespace_fallback(self):
        from section_a.a2_tokenizer import tokenize_sentence
        result = tokenize_sentence("राम ने खाना खाया")
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)


class TestA3UnigramFreq(unittest.TestCase):
    def test_build_unigram_freq(self):
        from section_a.a3_unigram_freq import build_unigram_freq
        from collections import Counter
        import tempfile, os

        content = "राम ने आम खाया\nराम गया\n"
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8",
                                         suffix=".txt", delete=False) as f:
            f.write(content)
            tmp = Path(f.name)

        freq = build_unigram_freq(tmp)
        tmp.unlink()
        self.assertEqual(freq["राम"], 2)
        self.assertEqual(freq["ने"], 1)


class TestA4BigramFreq(unittest.TestCase):
    def test_bigram_count(self):
        from section_a.a4_bigram_freq import build_bigram_freq
        import tempfile

        content = "राम ने आम खाया\n"
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8",
                                          suffix=".txt", delete=False) as f:
            f.write(content)
            tmp = Path(f.name)

        freq = build_bigram_freq(tmp)
        tmp.unlink()
        self.assertIn(("राम", "ने"), freq)
        self.assertEqual(freq[("राम", "ने")], 1)


class TestA6Stopwords(unittest.TestCase):
    @patch("section_a.a6_stopwords.ollama_chat", return_value="का, की, के, में, है")
    def test_llm_suggest(self, mock_llm):
        from section_a.a6_stopwords import llm_suggest_stopwords
        result = llm_suggest_stopwords()
        self.assertIn("का", result)
        self.assertGreater(len(result), 2)


class TestA9LM(unittest.TestCase):
    @patch("section_a.a9_ngram_lm.ollama_chat", return_value="8")
    def test_perplexity_range(self, mock_llm):
        from section_a.a9_ngram_lm import estimate_perplexity
        import math
        pp = estimate_perplexity("भारत एक महान देश है")
        # fluency=8 → pp = e^(10-8) = e^2 ≈ 7.39
        self.assertAlmostEqual(pp, math.exp(2), places=1)


if __name__ == "__main__":
    unittest.main()
