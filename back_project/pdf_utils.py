"""PDF text extraction and fixed-size chunking with optional overlap."""

from __future__ import annotations

import io
from pypdf import PdfReader

DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 75


class PdfExtractError(Exception):
    """Raised when PDF bytes cannot be read or parsed."""


def extract_text_from_pdf(data: bytes) -> str:
    """Extract plain text from PDF bytes. Raises PdfExtractError on failure."""
    try:
        reader = PdfReader(io.BytesIO(data))
    except Exception as e:
        raise PdfExtractError(f"Invalid PDF: {e}") from e

    parts: list[str] = []
    for page in reader.pages:
        parts.append(page.extract_text() or "")
    return "\n".join(parts).strip()


def split_into_chunks(
    text: str,
    size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    """
    Split text into chunks of at most ``size`` characters.
    Consecutive chunks overlap by ``overlap`` characters (capped so step stays positive).
    """
    if not text:
        return []
    if size <= 0:
        raise ValueError("chunk size must be positive")
    step = size - max(0, min(overlap, size - 1))
    chunks: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        chunks.append(text[i : i + size])
        i += step
        if step <= 0:
            break
    return chunks
