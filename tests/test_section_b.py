"""
tests/test_section_b.py
────────────────────────
Unit tests for Section B supervised models.
All LLM calls and model loads are mocked.
"""

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestB1POSTagger(unittest.TestCase):
    @patch("section_b.b1_pos_tagger.ollama_chat",
           return_value="राम/NN ने/PP आम/NN खाया/VB")
    def test_llm_pos_tag_returns_pairs(self, _):
        from section_b.b1_pos_tagger import llm_pos_tag
        result = llm_pos_tag("राम ने आम खाया")
        self.assertIsInstance(result, list)
        self.assertTrue(all(isinstance(t, tuple) and len(t) == 2 for t in result))
        words = [w for w, _ in result]
        self.assertIn("राम", words)

    @patch("section_b.b1_pos_tagger.ollama_chat",
           return_value="invalid response with no slashes")
    def test_llm_pos_tag_fallback(self, _):
        from section_b.b1_pos_tagger import llm_pos_tag
        result = llm_pos_tag("राम गया")
        # Fallback: all tagged NN
        self.assertTrue(all(t == "NN" for _, t in result))


class TestB2NERModel(unittest.TestCase):
    @patch("section_b.b2_ner_model.ollama_chat",
           return_value="B-PER I-PER B-LOC O O O")
    def test_llm_ner_tag_alignment(self, _):
        from section_b.b2_ner_model import llm_ner_tag
        result = llm_ner_tag("सचिन तेंदुलकर मुंबई में रहते हैं")
        self.assertEqual(len(result), 6)
        words = [w for w, _ in result]
        self.assertIn("सचिन", words)


class TestB3Sentiment(unittest.TestCase):
    @patch("section_b.b3_sentiment.ollama_chat", return_value="positive")
    def test_predict_llm_fallback(self, _):
        from section_b.b3_sentiment import predict
        # Model dir doesn't exist – uses LLM fallback
        result = predict("यह बहुत अच्छा है")
        self.assertIn(result["label"], ["positive", "negative", "neutral"])

    def test_label_map_complete(self):
        from section_b.b3_sentiment import LABEL_MAP, ID2LABEL
        self.assertEqual(len(LABEL_MAP), 3)
        self.assertEqual(set(LABEL_MAP.keys()), {"positive", "negative", "neutral"})


class TestB4TextClassifier(unittest.TestCase):
    @patch("section_b.b4_text_classifier.ollama_chat", return_value="politics")
    def test_classify_returns_known_topic(self, _):
        from section_b.b4_text_classifier import llm_classify
        topic, conf = llm_classify("सरकार ने नई नीति बनाई")
        self.assertIn(topic, ["news", "sports", "entertainment", "politics",
                               "science", "health", "education", "technology"])

    @patch("section_b.b4_text_classifier.ollama_chat", return_value="unknown_topic")
    def test_classify_unknown_falls_back(self, _):
        from section_b.b4_text_classifier import llm_classify
        topic, conf = llm_classify("कुछ भी")
        self.assertEqual(topic, "news")  # default fallback
        self.assertEqual(conf, 0.3)


class TestB5LangIdentifier(unittest.TestCase):
    def test_word_lang_hindi(self):
        from section_b.b5_lang_identifier import word_lang
        self.assertEqual(word_lang("भारत"), "HI")

    def test_word_lang_english(self):
        from section_b.b5_lang_identifier import word_lang
        self.assertEqual(word_lang("computer"), "EN")

    def test_detect_hindi_sentence(self):
        from section_b.b5_lang_identifier import detect
        result = detect("भारत एक महान देश है")
        self.assertIn(result["label"], ["HI", "MIX", "EN"])
        self.assertIn("word_labels", result)
        self.assertGreater(len(result["word_labels"]), 0)


if __name__ == "__main__":
    unittest.main()
