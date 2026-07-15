"""Ingest PDF into RAM store; query with hybrid retrieval + cross-encoder rerank + Ollama."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from embedding_model import EmbeddingModel
from hybrid_search import reciprocal_rank_fusion
from keyword_index import KeywordIndex
from ollama_client import chat_completion
from pdf_utils import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    extract_text_from_pdf,
    split_into_chunks,
)
from reranker import CrossEncoderReranker
from vector_store import ChromaVectorStore, SearchHit

# Per-method retrieval, RRF merge pool, and final LLM context (not exposed on API)
RETRIEVE_TOP_K = 30
MERGE_TOP_K = 22
CONTEXT_TOP_N = 6


@dataclass(frozen=True)
class IngestResult:
    filename: str
    chunk_count: int
    total_chars: int
    embedding_dim: int


@dataclass(frozen=True)
class RagContextItem:
    chunk_id: str
    chunk_index: int
    retrieval_score: float
    rerank_score: float
    text: str
    source: str | None = None
    retrieval_sources: str | None = None


@dataclass(frozen=True)
class RagQueryResult:
    answer: str
    vector_candidates: list[RagContextItem]
    keyword_candidates: list[RagContextItem]
    merged_candidates: list[RagContextItem]
    context_used: list[RagContextItem]
    llm_prompt: str


def _context_item(hit: SearchHit, *, rerank_score: float = 0.0) -> RagContextItem:
    return RagContextItem(
        chunk_id=hit.chunk_id,
        chunk_index=hit.chunk_index,
        retrieval_score=hit.score,
        rerank_score=rerank_score,
        text=hit.text,
        source=hit.source,
        retrieval_sources=hit.retrieval_sources,
    )


def ingest_pdf(
    store: ChromaVectorStore,
    keyword_index: KeywordIndex,
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
        chunk_ids: list[str] = []
    else:
        embeddings = embedder.encode(chunks)
        chunk_ids = store.add_chunks(chunks, embeddings, source=filename)

    if chunk_ids:
        keyword_index.add_chunks(chunk_ids, chunks, source=filename)

    return IngestResult(
        filename=filename,
        chunk_count=len(chunks),
        total_chars=len(text),
        embedding_dim=embedder.embedding_dim,
    )


def build_rag_user_prompt(question: str, hits: list[SearchHit]) -> str:
    parts: list[str] = [
        "The excerpts below were retrieved from uploaded PDF documents.",
        "They may be incomplete or cut between chunks.",
        "",
        "<documents>",
    ]
    for i, h in enumerate(hits, start=1):
        source = h.source or "unknown"
        parts.append(f'<document id="{i}" source="{source}">')
        parts.append(h.text)
        parts.append("</document>")
        parts.append("")
    parts.extend([
        "</documents>",
        "",
        f"Question: {question}",
        "",
        "Answer the question using only the excerpts above.",
        "Combine relevant excerpts into one clear, complete answer.",
        "Ignore excerpts that are not relevant to the question.",
        "In answer do not mention ids of the documents or chunks.",
        "If the excerpts do not contain enough information, reply exactly:",
        "I cannot find the answer in the provided documents.",
        "",
        "Answer:",
    ])
    return "\n".join(parts).strip()


def _empty_query_result(answer: str, *, llm_prompt: str) -> RagQueryResult:
    return RagQueryResult(
        answer=answer,
        vector_candidates=[],
        keyword_candidates=[],
        merged_candidates=[],
        context_used=[],
        llm_prompt=llm_prompt,
    )


async def query_rag(
    store: ChromaVectorStore,
    keyword_index: KeywordIndex,
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
                    "content": (
                        "You are a document Q&A assistant. "
                        "Use only the document excerpts in the user message. "
                        "Do not use outside knowledge."
                    ),
                },
                {"role": "user", "content": question},
            ],
            model=ollama_model,
        )
        return _empty_query_result(answer, llm_prompt=question)

    q_emb = embedder.encode_one(question)
    vector_hits = store.search(q_emb, top_k=RETRIEVE_TOP_K)
    keyword_hits = keyword_index.search(question, top_k=RETRIEVE_TOP_K)
    merged_hits = reciprocal_rank_fusion(
        vector_hits,
        keyword_hits,
        top_k=MERGE_TOP_K,
    )

    vector_candidates = [_context_item(h) for h in vector_hits]
    keyword_candidates = [_context_item(h) for h in keyword_hits]
    merged_candidates = [_context_item(h) for h in merged_hits]

    if not merged_hits:
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
        return RagQueryResult(
            answer=answer,
            vector_candidates=vector_candidates,
            keyword_candidates=keyword_candidates,
            merged_candidates=merged_candidates,
            context_used=[],
            llm_prompt=question,
        )

    passages = [h.text for h in merged_hits]
    ranked = reranker.rerank(question, passages, top_n=CONTEXT_TOP_N)
    rerank_by_chunk = {merged_hits[r.index].chunk_id: r.score for r in ranked}
    hits_ordered = [merged_hits[r.index] for r in ranked]

    user_content = build_rag_user_prompt(question, hits_ordered)
    messages = [
        {
            "role": "system",
            "content": "You are a precise assistant. Base your answer only on the provided context.",
        },
        {"role": "user", "content": user_content},
    ]
    answer = await chat_completion(messages, model=ollama_model)

    context_used = [
        _context_item(
            h,
            rerank_score=rerank_by_chunk[h.chunk_id],
        )
        for h in hits_ordered
    ]
    return RagQueryResult(
        answer=answer,
        vector_candidates=vector_candidates,
        keyword_candidates=keyword_candidates,
        merged_candidates=merged_candidates,
        context_used=context_used,
        llm_prompt=user_content,
    )
