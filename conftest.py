# Hindi NLP Toolkit – conftest.py
# Shared pytest fixtures and path setup

import sys
from pathlib import Path
import pytest

# Ensure project root is on sys.path for all tests
sys.path.insert(0, str(Path(__file__).parent))


@pytest.fixture(autouse=True)
def isolate_data_dir(tmp_path, monkeypatch):
    """
    Redirect all config data paths to a temp directory
    so tests never touch the real data/ folder.
    """
    import config
    monkeypatch.setattr(config, "DATA_DIR",           tmp_path / "data")
    monkeypatch.setattr(config, "MODELS_DIR",         tmp_path / "models")
    monkeypatch.setattr(config, "RAW_CORPUS",         tmp_path / "data/raw_corpus.txt")
    monkeypatch.setattr(config, "TOKENIZED_CORPUS",   tmp_path / "data/tokenized_corpus.txt")
    monkeypatch.setattr(config, "UNIGRAMS_JSON",      tmp_path / "data/unigrams.json")
    monkeypatch.setattr(config, "BIGRAMS_JSON",       tmp_path / "data/bigrams.json")
    monkeypatch.setattr(config, "TRIGRAMS_JSON",      tmp_path / "data/trigrams.json")
    monkeypatch.setattr(config, "STOPWORDS_TXT",      tmp_path / "data/stopwords.txt")
    monkeypatch.setattr(config, "EMBEDDINGS_DIR",     tmp_path / "data/embeddings")
    monkeypatch.setattr(config, "FAISS_INDEX",        tmp_path / "data/faiss_index")
    monkeypatch.setattr(config, "TRANSLITERATION_PAIRS", tmp_path / "data/transliteration_pairs.jsonl")
    (tmp_path / "data").mkdir(exist_ok=True)
    (tmp_path / "models").mkdir(exist_ok=True)
