"""
config.py – Centralized project configuration.
All modules import from here so settings stay consistent.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Indic NLP Resources (set before any indicnlp import) ──────────────────────
_INDIC_RES = os.getenv("INDIC_RESOURCES_PATH")
if _INDIC_RES:
    try:
        from indicnlp import common as _indic_common
        _indic_common.set_resources_path(_INDIC_RES)
    except Exception:
        pass  # falls back to whitespace tokenizer gracefully

# ── Project Paths ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# ── Corpus Paths ───────────────────────────────────────────────────────────────
WIKI_XML_PATH   = DATA_DIR / "hiwiki-20260201-pages-articles-multistream.xml.bz2"
RAW_CORPUS      = DATA_DIR / "raw_corpus.txt"
TOKENIZED_CORPUS= DATA_DIR / "tokenized_corpus.txt"
UNIGRAMS_JSON   = DATA_DIR / "unigrams.json"
BIGRAMS_JSON    = DATA_DIR / "bigrams.json"
TRIGRAMS_JSON   = DATA_DIR / "trigrams.json"
STOPWORDS_TXT   = DATA_DIR / "stopwords.txt"
EMBEDDINGS_DIR  = DATA_DIR / "embeddings"
FAISS_INDEX     = DATA_DIR / "faiss_index"
TRANSLITERATION_PAIRS = DATA_DIR / "transliteration_pairs.jsonl"

# ── Ollama LLM Configuration ────────────────────────────────────────────────────
OLLAMA_BASE_URL  = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL     = os.getenv("OLLAMA_MODEL", "qwen2:0.5b")
OLLAMA_CHAT_ENDPOINT = f"{OLLAMA_BASE_URL}/api/chat"

# ── Embedding / Transformer Models ─────────────────────────────────────────────
SENTENCE_TRANSFORMER_MODEL = os.getenv("SENTENCE_TRANSFORMER_MODEL", "sentence-transformers/LaBSE")
WORD_EMBEDDING_MODEL       = os.getenv("WORD_EMBEDDING_MODEL", "xlm-roberta-base")
MT5_MODEL                  = os.getenv("MT5_MODEL", "google/mt5-small")
XLMR_MODEL                 = os.getenv("XLMR_MODEL", "xlm-roberta-base")
INDICBERT_MODEL            = os.getenv("INDICBERT_MODEL", "xlm-roberta-base")

# ── Training Hyperparameters ────────────────────────────────────────────────────
BATCH_SIZE           = int(os.getenv("BATCH_SIZE", "16"))
LEARNING_RATE        = float(os.getenv("LEARNING_RATE", "2e-5"))
NUM_TRAIN_EPOCHS     = int(os.getenv("NUM_TRAIN_EPOCHS", "3"))
MAX_SEQ_LENGTH       = int(os.getenv("MAX_SEQ_LENGTH", "128"))
SYNTHETIC_SENT_COUNT = int(os.getenv("SYNTHETIC_SENT_COUNT", "500"))

# ── API Keys (optional – for future cloud LLM fallback) ────────────────────────
OPENWEATHER_API_KEY  = os.getenv("OPENWEATHER_API_KEY", "81ab662082b3b2bf5b192d6313f5bcbe")

# ── FastAPI / Streamlit ─────────────────────────────────────────────────────────
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
