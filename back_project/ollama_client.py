"""Async client for Ollama chat API."""

from __future__ import annotations

import os
from typing import Any

import httpx

DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_MODEL = "phi3"


def _base_url() -> str:
    return os.environ.get("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL).rstrip("/")


def _default_model() -> str:
    return os.environ.get("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)


def _http_timeout() -> httpx.Timeout:
    """
    Ollama generation can take minutes on CPU; default read timeout is generous.
    Override with OLLAMA_TIMEOUT (seconds), e.g. 900.
    """
    read_s = float(os.environ.get("OLLAMA_TIMEOUT", "600"))
    return httpx.Timeout(connect=30.0, read=read_s, write=120.0, pool=10.0)


async def chat_completion(
    messages: list[dict[str, str]],
    *,
    model: str | None = None,
) -> str:
    """
    Call Ollama ``/api/chat`` (non-streaming). Returns assistant message content.
    """
    url = f"{_base_url()}/api/chat"
    payload: dict[str, Any] = {
        "model": model or _default_model(),
        "messages": messages,
        "stream": False,
    }
    async with httpx.AsyncClient(timeout=_http_timeout()) as client:
        r = await client.post(url, json=payload)
        r.raise_for_status()
        data = r.json()
    msg = data.get("message") or {}
    content = msg.get("content")
    if not isinstance(content, str):
        raise RuntimeError("Ollama response missing message.content")
    return content
