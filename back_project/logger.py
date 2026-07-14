"""Application logging: one dated log file per process start."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Protocol

_LOG_DIR = Path(__file__).resolve().parent / "logs"
_logger: logging.Logger | None = None


class RagChunkLog(Protocol):
    chunk_id: str
    chunk_index: int
    retrieval_score: float
    rerank_score: float
    text: str
    source: str | None


def setup_logging() -> logging.Logger:
    """Configure logging once per process; file name includes start date and time."""
    global _logger
    if _logger is not None:
        return _logger

    _LOG_DIR.mkdir(exist_ok=True)
    started_at = datetime.now()
    log_file = _LOG_DIR / f"rag_{started_at:%Y-%m-%d_%H-%M-%S}.log"

    logger = logging.getLogger("diplomski")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logger.info("Logging started; file=%s", log_file)
    _logger = logger
    return logger


def get_logger() -> logging.Logger:
    if _logger is None:
        return setup_logging()
    return _logger


def _log_chunk_entries(
    logger: logging.Logger,
    title: str,
    chunks: list[RagChunkLog],
    *,
    show_rerank: bool,
) -> None:
    if not chunks:
        logger.info("%s: none", title)
        return

    logger.info("%s (%d):", title, len(chunks))
    for i, chunk in enumerate(chunks, start=1):
        if show_rerank:
            logger.info(
                "  [%d] chunk_id=%s chunk_index=%s source=%s "
                "retrieval_score=%.4f rerank_score=%.4f",
                i,
                chunk.chunk_id,
                chunk.chunk_index,
                chunk.source or "unknown",
                chunk.retrieval_score,
                chunk.rerank_score,
            )
        else:
            logger.info(
                "  [%d] chunk_id=%s chunk_index=%s source=%s retrieval_score=%.4f",
                i,
                chunk.chunk_id,
                chunk.chunk_index,
                chunk.source or "unknown",
                chunk.retrieval_score,
            )
        logger.info("      text: %s", chunk.text.replace("\n", " "))


def log_rag_query(
    question: str,
    ranked_chunks: list[RagChunkLog],
    *,
    retrieved_chunks: list[RagChunkLog] | None = None,
    llm_prompt: str | None = None,
) -> None:
    logger = get_logger()
    logger.info("Question: %s", question)

    if retrieved_chunks is not None:
        _log_chunk_entries(
            logger,
            "Vector retrieval candidates",
            retrieved_chunks,
            show_rerank=False,
        )

    _log_chunk_entries(
        logger,
        "Top ranked chunks sent to LLM",
        ranked_chunks,
        show_rerank=True,
    )

    if llm_prompt is not None:
        logger.info("LLM user prompt:\n%s", llm_prompt)
