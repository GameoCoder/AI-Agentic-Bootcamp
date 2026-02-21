"""
utils/ollama_client.py
──────────────────────
Thin wrapper around the Ollama /api/chat endpoint for qwen2:0.5b.
All LLM calls in the project route through this module.
"""

import json
import sys
from pathlib import Path
import requests

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


def ollama_chat(
    prompt: str,
    system: str = "तुम एक उपयोगी हिन्दी भाषा सहायक हो।",
    model: str | None = None,
    temperature: float = 0.3,
    stream: bool = False,
) -> str:
    """
    Send a chat message to Ollama and return the assistant reply as a string.

    Args:
        prompt:      User message.
        system:      System prompt (defaults to Hindi assistant persona).
        model:       Model name (defaults to config.OLLAMA_MODEL).
        temperature: Sampling temperature.
        stream:      If True, streams and concatenates chunks.

    Returns:
        The assistant reply text.

    Raises:
        RuntimeError: If Ollama is unreachable or returns an error.
    """
    model = model or config.OLLAMA_MODEL
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt},
        ],
        "options": {"temperature": temperature},
        "stream": stream,
    }

    try:
        resp = requests.post(
            config.OLLAMA_CHAT_ENDPOINT,
            json=payload,
            timeout=300,
            stream=stream,
        )
        resp.raise_for_status()
    except requests.exceptions.ConnectionError as e:
        raise RuntimeError(
            f"Cannot reach Ollama at {config.OLLAMA_CHAT_ENDPOINT}. "
            "Is Ollama running? (ollama serve)"
        ) from e

    if stream:
        chunks = []
        for line in resp.iter_lines():
            if line:
                data = json.loads(line)
                msg = data.get("message", {}).get("content", "")
                chunks.append(msg)
                if data.get("done"):
                    break
        return "".join(chunks)
    else:
        data = resp.json()
        return data.get("message", {}).get("content", "")


def ollama_generate_list(
    prompt: str,
    system: str = "तुम एक उपयोगी हिन्दी भाषा सहायक हो।",
    separator: str = "\n",
) -> list[str]:
    """Call LLM and split response into a list by separator."""
    raw = ollama_chat(prompt, system=system)
    return [line.strip() for line in raw.split(separator) if line.strip()]
