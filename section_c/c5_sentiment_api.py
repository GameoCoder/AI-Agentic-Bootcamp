"""
c5_sentiment_api.py
───────────────────
FastAPI REST endpoint wrapping the Hindi sentiment model (b3_sentiment).

Endpoints:
  POST /analyze        – Analyze sentiment of text
  GET  /health         – Health check
  POST /batch          – Analyze list of texts

Usage:
    uvicorn section_c.c5_sentiment_api:app --reload --port 8000
    # or
    python section_c/c5_sentiment_api.py
"""

import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import config

# Lazy-load sentiment model
_sentiment_predict = None


def get_sentiment_predict():
    global _sentiment_predict
    if _sentiment_predict is None:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from section_b.b3_sentiment import predict
        _sentiment_predict = predict
    return _sentiment_predict


# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Hindi Sentiment Analysis API",
    description="Analyzes sentiment (positive/negative/neutral) of Hindi text.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ────────────────────────────────────────────────────────────────────

class TextInput(BaseModel):
    text: str
    model_config = {"json_schema_extra": {"example": {"text": "यह फिल्म बहुत अच्छी थी"}}}


class BatchInput(BaseModel):
    texts: list[str]


class SentimentResult(BaseModel):
    text: str
    label: str
    score: float
    source: Optional[str] = None


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "model": config.OLLAMA_MODEL}


@app.post("/analyze", response_model=SentimentResult)
def analyze(body: TextInput):
    if not body.text.strip():
        raise HTTPException(status_code=422, detail="text cannot be empty")
    predict = get_sentiment_predict()
    result = predict(body.text)
    return SentimentResult(text=body.text, **result)


@app.post("/batch", response_model=list[SentimentResult])
def batch_analyze(body: BatchInput):
    if not body.texts:
        raise HTTPException(status_code=422, detail="texts list is empty")
    predict = get_sentiment_predict()
    results = []
    for text in body.texts:
        res = predict(text)
        results.append(SentimentResult(text=text, **res))
    return results


if __name__ == "__main__":
    import uvicorn as _uvicorn
    _uvicorn.run("section_c.c5_sentiment_api:app", host=config.API_HOST, port=config.API_PORT, reload=True)
