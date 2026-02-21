# Hindi NLP Toolkit

A comprehensive Hindi NLP project built as agentic pipelines using **LangChain / LangGraph** + **Ollama (qwen2:0.5b)**.

## 🚀 Quick Start

```bash
# 1. Clone and enter
cd AI-Agentic-Bootcamp

# 2. Install dependencies
pip install -r requirements.txt

# 3. Copy and configure environment
cp .env.example .env
# Make sure Ollama is running: ollama serve
# Make sure qwen2:0.5b is pulled: ollama pull qwen2:0.5b

# 4. Run the data pipeline first
python section_a/a1_corpus_crawler.py    # Extract from hiwiki XML
python section_a/a2_tokenizer.py
python section_a/a3_unigram_freq.py
python section_a/a4_bigram_freq.py
python section_a/a5_trigram_freq.py
python section_a/a6_stopwords.py
python section_a/a8_sentence_embeddings.py   # Builds FAISS index

# 5. Run tests
pytest tests/ -v
```

---

## 📁 Project Structure

```
AI-Agentic-Bootcamp/
├── config.py                   # Centralized config (Ollama URL, paths)
├── requirements.txt
├── .env.example
├── conftest.py                 # Pytest fixtures
│
├── utils/
│   └── ollama_client.py        # Ollama qwen2:0.5b wrapper
│
├── section_a/                  # Datasets & Language Resources (A1–A11)
├── section_b/                  # Supervised Models (B1–B5)
├── section_c/                  # Tools & Applications (C1–C9)
├── section_d/                  # Evaluation & Analysis (D1–D2)
│
├── data/                       # Generated corpus (gitignored)
├── models/                     # Trained models (gitignored)
└── tests/                      # Unit tests (A/B/C/D)
```

---

## 🗂️ Section Overview

### A – Datasets & Language Resources

| Module | Description |
|---|---|
| `a1_corpus_crawler.py` | Extracts Hindi sentences from `hiwiki` XML dump + LLM filtering + synthetic generation |
| `a2_tokenizer.py` | Tokenizes corpus using Indic NLP Library |
| `a3_unigram_freq.py` | Word frequency counter → `data/unigrams.json` |
| `a4_bigram_freq.py` | Bigram frequency counter + LLM quality filter |
| `a5_trigram_freq.py` | Trigram frequency counter + LLM validation |
| `a6_stopwords.py` | LLM + frequency-based stop word generation |
| `a7_word_embeddings.py` | IndicBERT contextual word embeddings |
| `a8_sentence_embeddings.py` | LaBSE sentence embeddings + FAISS index |
| `a9_ngram_lm.py` | LLM-based language model (pseudo-perplexity) |
| `a10_morphological.py` | LLM morpheme analysis → JSONL training data |
| `a11_transliteration_pairs.py` | Roman↔Devanagari pairs via LLM + Wikipedia crawl |

### B – Supervised Models

| Module | Description |
|---|---|
| `b1_pos_tagger.py` | LLM silver POS labels → XLM-R fine-tuning |
| `b2_ner_model.py` | LLM BIO NER labels → XLM-R token classifier |
| `b3_sentiment.py` | LLM synthetic data → IndicBERT 3-class sentiment |
| `b4_text_classifier.py` | Zero-shot LLM topic labels → DistilBERT classifier |
| `b5_lang_identifier.py` | Code-mixed data → char n-gram SGD classifier |

### C – Tools & Applications

| Module | How to Run |
|---|---|
| `c1_chatbot.py` | `python section_c/c1_chatbot.py` |
| `c2_rag_system.py` | `python section_c/c2_rag_system.py` |
| `c3_summarizer.py` | `python section_c/c3_summarizer.py --text "..."` |
| `c4_spell_checker.py` | `python section_c/c4_spell_checker.py --text "..."` |
| `c5_sentiment_api.py` | `uvicorn section_c.c5_sentiment_api:app --reload` |
| `c6_pos_ner_demo.py` | `streamlit run section_c/c6_pos_ner_demo.py` |
| `c7_word_cloud.py` | `streamlit run section_c/c7_word_cloud.py` |
| `c8_lang_detector.py` | `python section_c/c8_lang_detector.py --text "..."` |
| `c9_transliteration_tool.py` | `streamlit run section_c/c9_transliteration_tool.py` |

### D – Evaluation & Analysis

| Module | How to Run |
|---|---|
| `d1_readability_analyzer.py` | `streamlit run section_d/d1_readability_analyzer.py` |
| `d2_corpus_stats_dashboard.py` | `streamlit run section_d/d2_corpus_stats_dashboard.py` |

---

## ⚙️ Configuration

All settings live in `config.py` and are loaded from `.env`:

| Variable | Default | Purpose |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `qwen2:0.5b` | LLM model name |
| `BATCH_SIZE` | `16` | Training batch size |
| `NUM_TRAIN_EPOCHS` | `3` | Fine-tuning epochs |
| `SYNTHETIC_SENT_COUNT` | `500` | Synthetic sentences to generate |

---

## 📊 Data Flow

```
hiwiki XML dump (data/)
        │
        ▼
  A1: Extract Hindi sentences
        │
        ▼
  A2: Tokenize (Indic NLP)
        │
   ┌────┴────┐
   ▼         ▼
 A3–A5:     A6:
 N-gram     Stopwords
 Counts         │
   │            ▼
   │       A7–A8:
   │       Embeddings + FAISS
   │            │
   └─────┬──────┘
         ▼
    B1–B5: Supervised Models
         │
         ▼
    C1–C9: Applications & APIs
         │
         ▼
    D1–D2: Evaluation Dashboards
```

---

## 🧪 Testing

```bash
pytest tests/ -v                    # all tests
pytest tests/test_section_a.py -v  # Section A only
pytest tests/test_section_b.py -v  # Section B only
pytest tests/test_section_c.py -v  # Section C only
pytest tests/test_section_d.py -v  # Section D only
```

Tests use `unittest.mock` to avoid actual LLM/network calls. The `conftest.py` redirects all data paths to a temp directory.

---

## 📋 Prerequisites

- Python ≥ 3.10
- [Ollama](https://ollama.ai) running locally with `qwen2:0.5b` pulled
- Hindi Wikipedia XML dump at `data/hiwiki-20260201-pages-articles-multistream.xml.bz2`
- GPU optional (training steps work on CPU but are slower)