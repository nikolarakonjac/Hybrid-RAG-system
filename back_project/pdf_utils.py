"""PDF text extraction and sentence-aware chunking with optional overlap."""

from __future__ import annotations

import re

import fitz

DEFAULT_CHUNK_SIZE = 1600
DEFAULT_CHUNK_OVERLAP = 200

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


class PdfExtractError(Exception):
    """Raised when PDF bytes cannot be read or parsed."""


def _normalize_whitespace(text: str) -> str:
    """Collapse runs of spaces/tabs and single line breaks within a text block."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n+", " ", text).strip()


def extract_text_from_pdf(data: bytes) -> str:
    """Extract plain text from PDF bytes. Raises PdfExtractError on failure."""
    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception as e:
        raise PdfExtractError(f"Invalid PDF: {e}") from e

    try:
        blocks: list[str] = []
        for page in doc:
            for block in page.get_text("blocks"):
                if len(block) < 7 or block[6] != 0:
                    continue
                normalized = _normalize_whitespace(block[4])
                if normalized:
                    blocks.append(normalized)

        if blocks:
            return "\n\n".join(blocks)

        # Fallback when block extraction yields nothing (unusual PDFs).
        parts: list[str] = []
        for page in doc:
            normalized = _normalize_whitespace(page.get_text() or "")
            if normalized:
                parts.append(normalized)
        return "\n\n".join(parts)
    finally:
        doc.close()


def _split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    return [part.strip() for part in _SENTENCE_SPLIT.split(text) if part.strip()]


def _split_by_characters(text: str, size: int, overlap: int) -> list[str]:
    """Fallback splitter for text without reliable sentence boundaries."""
    step = size - max(0, min(overlap, size - 1))
    if step <= 0:
        raise ValueError("overlap must be smaller than size")

    chunks: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        end = min(i + size, n)
        chunk = text[i:end]
        if end < n and " " in chunk:
            trimmed = chunk.rsplit(" ", 1)[0]
            if trimmed:
                chunk = trimmed
        chunks.append(chunk)
        if end >= n:
            break
        i += max(len(chunk) - overlap, 1)
    return chunks


def _chunks_from_sentences(
    sentences: list[str],
    size: int,
    overlap: int,
) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for sentence in sentences:
        added_len = len(sentence) + (1 if current else 0)
        if current and current_len + added_len > size:
            chunks.append(" ".join(current))

            overlap_sents: list[str] = []
            overlap_len = 0
            for sent in reversed(current):
                need = len(sent) + (1 if overlap_sents else 0)
                if overlap_len + need > overlap:
                    break
                overlap_sents.insert(0, sent)
                overlap_len += need

            current = overlap_sents[:]
            current_len = overlap_len

        current.append(sentence)
        current_len += added_len

    if current:
        chunks.append(" ".join(current))
    return chunks


def split_into_chunks(
    text: str,
    size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    """
    Split text into chunks of at most ``size`` characters.

    Short paragraphs are merged up to ``size``. Longer paragraphs are split on
    sentence boundaries with ``overlap`` characters of trailing sentences carried
    into the next chunk.
    """
    if not text:
        return []
    if size <= 0:
        raise ValueError("chunk size must be positive")

    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    if not paragraphs:
        return []

    chunks: list[str] = []
    batch: list[str] = []
    batch_len = 0

    def flush_batch() -> None:
        nonlocal batch, batch_len
        if batch:
            chunks.append("\n\n".join(batch))
            batch = []
            batch_len = 0

    for paragraph in paragraphs:
        paragraph = re.sub(r"\s+", " ", paragraph).strip()
        if not paragraph:
            continue

        if len(paragraph) > size:
            flush_batch()
            sentences = _split_sentences(paragraph)
            if len(sentences) <= 1:
                chunks.extend(_split_by_characters(paragraph, size, overlap))
            else:
                chunks.extend(_chunks_from_sentences(sentences, size, overlap))
            continue

        separator_len = 2 if batch else 0
        if batch and batch_len + separator_len + len(paragraph) > size:
            flush_batch()

        if batch:
            batch_len += separator_len + len(paragraph)
        else:
            batch_len = len(paragraph)
        batch.append(paragraph)

    flush_batch()
    return chunks
