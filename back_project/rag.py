"""Ingest PDF into RAM store; query with bi-encoder retrieval + cross-encoder rerank + Ollama."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from embedding_model import EmbeddingModel
from ollama_client import chat_completion
from pdf_utils import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    extract_text_from_pdf,
    split_into_chunks,
)
from reranker import CrossEncoderReranker
from vector_store import ChromaVectorStore, SearchHit

# Fixed MVP retrieval / context sizes (not exposed on API)
RETRIEVE_TOP_K = 40
CONTEXT_TOP_N = 8


@dataclass(frozen=True)
class IngestResult:
    filename: str
    chunk_count: int
    total_chars: int
    embedding_dim: int


def ingest_pdf(
    store: ChromaVectorStore,
    embedder: EmbeddingModel,
    pdf_bytes: bytes,
    *,
    filename: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> IngestResult:
    text = extract_text_from_pdf(pdf_bytes)
    chunks = split_into_chunks(text, size=chunk_size, overlap=chunk_overlap)
    if not chunks:
        embeddings = np.zeros((0, embedder.embedding_dim), dtype=np.float32)
    else:
        embeddings = embedder.encode(chunks)
    store.add_chunks(chunks, embeddings, source=filename)
    return IngestResult(
        filename=filename,
        chunk_count=len(chunks),
        total_chars=len(text),
        embedding_dim=embedder.embedding_dim,
    )


def build_rag_user_prompt(question: str, hits: list[SearchHit]) -> str:
    parts: list[str] = [
        "Answer using only the context below. If the context does not contain enough information, say so.",
        "",
        "Context:",
    ]
    for i, h in enumerate(hits, start=1):
        parts.append(f"[{i}] (chunk_index={h.chunk_index})\n{h.text}")
        parts.append("")
    parts.append(f"Question: {question}")
    return "\n".join(parts).strip()


@dataclass(frozen=True)
class RagContextItem:
    chunk_id: str
    chunk_index: int
    retrieval_score: float
    rerank_score: float
    text: str
    source: str | None = None


@dataclass(frozen=True)
class RagQueryResult:
    answer: str
    context_used: list[RagContextItem]


async def query_rag(
    store: ChromaVectorStore,
    embedder: EmbeddingModel,
    reranker: CrossEncoderReranker,
    question: str,
    *,
    ollama_model: str | None = None,
) -> RagQueryResult:
    if not store.has_document():
        answer = await chat_completion(
            [
                {
                    "role": "system",
                    "content": "You are a helpful assistant. No document has been uploaded yet. Tell the user to upload a PDF first.",
                },
                {"role": "user", "content": question},
            ],
            model=ollama_model,
        )
        return RagQueryResult(answer=answer, context_used=[])

    q_emb = embedder.encode_one(question)
    hits = store.search(q_emb, top_k=RETRIEVE_TOP_K)
    if not hits:
        answer = await chat_completion(
            [
                {
                    "role": "system",
                    "content": "You are a helpful assistant. The document has no searchable text. Explain briefly that the PDF may be empty or scanned.",
                },
                {"role": "user", "content": question},
            ],
            model=ollama_model,
        )
        return RagQueryResult(answer=answer, context_used=[])

    passages = [h.text for h in hits]
    ranked = reranker.rerank(question, passages, top_n=CONTEXT_TOP_N)
    rerank_by_chunk = {hits[r.index].chunk_id: r.score for r in ranked}
    hits_ordered = [hits[r.index] for r in ranked]

    user_content = build_rag_user_prompt(question, hits_ordered)
    messages = [
        {
            "role": "system",
            "content": "You are a precise assistant. Base your answer only on the provided context.",
        },
        {"role": "user", "content": user_content},
    ]
    answer = await chat_completion(messages, model=ollama_model)

    context_used: list[RagContextItem] = []
    for h in hits_ordered:
        context_used.append(
            RagContextItem(
                chunk_id=h.chunk_id,
                chunk_index=h.chunk_index,
                retrieval_score=h.score,
                rerank_score=rerank_by_chunk[h.chunk_id],
                text=h.text,
                source=h.source,
            )
        )
    return RagQueryResult(answer=answer, context_used=context_used)
