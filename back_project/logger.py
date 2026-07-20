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
    retrieval_sources: str | None


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


def log_rag_query(
    question: str,
    llm_chunks: list[RagChunkLog],
    *,
    answer: str | None = None,
) -> None:
    """Log the question, chunks sent to the LLM, and the final answer."""
    logger = get_logger()
    logger.info("Question: %s", question)

    if answer is not None:
        logger.info("Answer: %s", answer.replace("\n", " "))

    if not llm_chunks:
        logger.info("Top ranked chunks sent to LLM: none")
        return

    logger.info("Top ranked chunks sent to LLM (%d):", len(llm_chunks))
    for i, chunk in enumerate(llm_chunks, start=1):
        sources = chunk.retrieval_sources or "unknown"
        logger.info(
            "  [%d] chunk_id=%s chunk_index=%s source=%s sources=%s "
            "rrf_score=%.4f rerank_score=%.4f",
            i,
            chunk.chunk_id,
            chunk.chunk_index,
            chunk.source or "unknown",
            sources,
            chunk.retrieval_score,
            chunk.rerank_score,
        )
        logger.info("      text: %s", chunk.text.replace("\n", " "))
