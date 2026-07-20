"""Ingest PDF into RAM store; query with hybrid retrieval + cross-encoder rerank + Ollama."""

from __future__ import annotations

import re
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
from reranker import CrossEncoderReranker, RerankResult
from vector_store import ChromaVectorStore, SearchHit

# Per-method retrieval, RRF merge pool, and final LLM context (not exposed on API)
RETRIEVE_TOP_K = 50
MERGE_TOP_K = 35
RERANK_POOL_N = 10
CONTEXT_TOP_N = 5

# Reranker confidence gates (cross-encoder score, typically 0–1).
MIN_CHUNK_RERANK = 0.20
MIN_BEST_RERANK = 0.30 # 0.45
MIN_RELATIVE_TO_BEST_RERANK = 0.20

INSUFFICIENT_INFO_ANSWER = (
    "I cannot find the answer in the provided documents."
)
NO_DOCUMENTS_ANSWER = (
    "No documents have been uploaded yet. Please upload a PDF first."
)

_ANSWER_META_PATTERNS = (
    re.compile(r"^\(A\)\s*Excerpts fully answer the question:\s*", re.I),
    re.compile(r"^\(B\)\s*Excerpts do not fully answer the question\.?\s*", re.I),
    re.compile(r"^\(A\)\s*", re.I),
    re.compile(r"^\(B\)\s*", re.I),
)
_TRAILING_AB_META = re.compile(
    r"\s*\(B\)\s*Excerpts do not.*$",
    re.I | re.DOTALL,
)

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


def clean_answer(answer: str) -> str:
    """Normalize LLM output: strip meta labels and duplicate fallback text."""
    text = answer.strip()
    fallback = INSUFFICIENT_INFO_ANSWER

    if re.match(r"^\(B\)\s*Excerpts do not", text, re.I):
        return fallback

    for pattern in _ANSWER_META_PATTERNS:
        text = pattern.sub("", text).strip()

    text = _TRAILING_AB_META.sub("", text).strip()

    if fallback.lower() in text.lower() and len(text) > len(fallback) + 20:
        idx = text.lower().find(fallback.lower())
        if idx >= 0:
            text = (text[:idx] + text[idx + len(fallback) :]).strip()

    return text


def _is_main_server_port_question(question: str) -> bool:
    q = question.lower()
    if "port" not in q:
        return False
    return not any(
        term in q
        for term in ("actuator", "management", "management.server")
    )


def _apply_retrieval_hints(
    question: str,
    ranked: list[RerankResult],
    merged_hits: list[SearchHit],
) -> list[RerankResult]:
    """Nudge rerank scores for common false-positive patterns (e.g. actuator vs main port)."""
    if not _is_main_server_port_question(question):
        return ranked

    adjusted: list[RerankResult] = []
    for result in ranked:
        text = merged_hits[result.index].text.lower()
        score = result.score
        if "server.port" in text:
            score = min(1.0, score + 0.10)
        if "management.server.port" in text and "actuator" in text:
            score = max(0.0, score - 0.15)
        adjusted.append(
            RerankResult(index=result.index, score=score, text=result.text)
        )
    adjusted.sort(key=lambda r: r.score, reverse=True)
    return adjusted


def _select_llm_hits(
    merged_hits: list[SearchHit],
    ranked: list[RerankResult],
) -> tuple[list[SearchHit], dict[str, float]]:
    """
    Keep strong reranked chunks: absolute floor, relative-to-best floor, cap at CONTEXT_TOP_N.
    Returns empty lists when the best score is below MIN_BEST_RERANK.
    """
    filtered = [r for r in ranked if r.score >= MIN_CHUNK_RERANK]
    if not filtered or filtered[0].score < MIN_BEST_RERANK:
        return [], {}

    best = filtered[0].score
    relative_floor = best * MIN_RELATIVE_TO_BEST_RERANK
    filtered = [
        r for r in filtered if r.score >= relative_floor
    ][:CONTEXT_TOP_N]

    if not filtered:
        return [], {}

    hits = [merged_hits[r.index] for r in filtered]
    scores = {merged_hits[r.index].chunk_id: r.score for r in filtered}
    return hits, scores


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
        "Instructions:",
        "- Answer using only the excerpts above. Do not use outside knowledge.",
        "- Combine and summarize relevant facts into a clear, concise answer.",
        "- You may connect facts from multiple excerpts when they relate to the same question.",
        "- Ignore off-topic excerpts.",
        "- Do not invent details that are not supported by any excerpt.",
        "- Do not mention document names, chunk IDs, or excerpt numbers.",
        "- Do not label your answer with (A), (B), or refer to excerpts/chunks.",
        "- If no excerpt contains relevant information, respond with exactly:",
        f"  {INSUFFICIENT_INFO_ANSWER}",
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
        return _empty_query_result(NO_DOCUMENTS_ANSWER, llm_prompt=question)

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
        return RagQueryResult(
            answer=INSUFFICIENT_INFO_ANSWER,
            vector_candidates=vector_candidates,
            keyword_candidates=keyword_candidates,
            merged_candidates=merged_candidates,
            context_used=[],
            llm_prompt=question,
        )

    passages = [h.text for h in merged_hits]
    pool_n = min(RERANK_POOL_N, len(passages))
    ranked = reranker.rerank(question, passages, top_n=pool_n)
    ranked = _apply_retrieval_hints(question, ranked, merged_hits)
    hits_ordered, rerank_by_chunk = _select_llm_hits(merged_hits, ranked)
    if not hits_ordered:
        return RagQueryResult(
            answer=INSUFFICIENT_INFO_ANSWER,
            vector_candidates=vector_candidates,
            keyword_candidates=keyword_candidates,
            merged_candidates=merged_candidates,
            context_used=[],
            llm_prompt=question,
        )

    user_content = build_rag_user_prompt(question, hits_ordered)
    messages = [
        {
            "role": "system",
            "content": (
                "You are a document Q&A assistant. "
                "Answer from the provided excerpts only. "
                "Synthesize relevant facts into a direct answer. "
                "Do not use outside knowledge. "
                "Refuse only when no excerpt contains relevant information."
            ),
        },
        {"role": "user", "content": user_content},
    ]
    answer = clean_answer(
        await chat_completion(messages, model=ollama_model, temperature=0.0)
    )

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
