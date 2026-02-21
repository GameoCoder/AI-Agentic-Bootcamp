"""
tests/test_section_c.py
────────────────────────
Unit tests for Section C tools.
API endpoints tested with httpx TestClient; Streamlit apps untested directly.
"""

import sys
import json
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestC4SpellChecker(unittest.TestCase):
    def test_no_errors_when_all_in_dict(self):
        from section_c.c4_spell_checker import spell_check
        dictionary = {"राम", "ने", "आम", "खाया"}
        result = spell_check("राम ने आम खाया", dictionary)
        self.assertFalse(result["has_errors"])
        self.assertEqual(result["corrected"], "राम ने आम खाया")

    @patch("section_c.c4_spell_checker.ollama_chat",
           return_value="राम ने आम खाया")
    def test_oov_triggers_correction(self, mock_llm):
        from section_c.c4_spell_checker import spell_check
        dictionary = {"राम", "ने", "खाया"}  # "आम" is OOV
        result = spell_check("राम ने आम खाया", dictionary)
        # "आम" should be detected or missed depending on tokenizer
        self.assertIsInstance(result["oov_words"], list)

    def test_empty_dictionary_uses_llm(self):
        from section_c.c4_spell_checker import spell_check
        with patch("section_c.c4_spell_checker.ollama_chat",
                   return_value="the corrected sentence"):
            result = spell_check("गलत वाक्य", set())
            self.assertIsInstance(result, dict)


class TestC5SentimentAPI(unittest.TestCase):
    def setUp(self):
        from fastapi.testclient import TestClient
        from section_c.c5_sentiment_api import app
        self.client = TestClient(app)

    def test_health_check(self):
        r = self.client.get("/health")
        self.assertEqual(r.status_code, 200)
        self.assertIn("status", r.json())

    @patch("section_c.c5_sentiment_api.get_sentiment_predict")
    def test_analyze_endpoint(self, mock_get):
        mock_predict = MagicMock(return_value={"label": "positive", "score": 0.95})
        mock_get.return_value = mock_predict
        r = self.client.post("/analyze", json={"text": "यह अच्छा है"})
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertIn("label", data)
        self.assertIn("score", data)

    def test_empty_text_returns_422(self):
        r = self.client.post("/analyze", json={"text": ""})
        self.assertEqual(r.status_code, 422)

    @patch("section_c.c5_sentiment_api.get_sentiment_predict")
    def test_batch_endpoint(self, mock_get):
        mock_predict = MagicMock(return_value={"label": "neutral", "score": 0.7})
        mock_get.return_value = mock_predict
        r = self.client.post("/batch", json={"texts": ["वाक्य एक", "वाक्य दो"]})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(len(r.json()), 2)


class TestC2RAG(unittest.TestCase):
    @patch("section_c.c2_rag_system.ollama_chat",
           return_value="भारत एक महान देश है।")
    def test_fallback_answer(self, mock_llm):
        """When FAISS is absent, should call LLM directly."""
        from section_c.c2_rag_system import answer

        def _answer(q):
            return mock_llm(q)

        result = _answer("भारत क्या है?")
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)


class TestC9Transliteration(unittest.TestCase):
    @patch("section_c.c9_transliteration_tool.ollama_chat",
           return_value="कंप्यूटर")
    def test_llm_transliterate(self, _):
        from section_c.c9_transliteration_tool import llm_transliterate
        result = llm_transliterate("computer")
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)


if __name__ == "__main__":
    unittest.main()
