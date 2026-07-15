"""Merge vector and keyword retrieval results with reciprocal rank fusion (RRF)."""

from __future__ import annotations

from vector_store import SearchHit

RRF_K = 60


def reciprocal_rank_fusion(
    vector_hits: list[SearchHit],
    keyword_hits: list[SearchHit],
    *,
    top_k: int,
    rrf_k: int = RRF_K,
) -> list[SearchHit]:
    """Combine ranked lists by rank position; raw scores are not compared directly."""
    if top_k <= 0:
        return []

    scores: dict[str, float] = {}
    by_id: dict[str, SearchHit] = {}
    sources_by_id: dict[str, set[str]] = {}

    for label, hits in (("vector", vector_hits), ("keyword", keyword_hits)):
        for rank, hit in enumerate(hits):
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1.0 / (
                rrf_k + rank + 1
            )
            by_id[hit.chunk_id] = hit
            sources_by_id.setdefault(hit.chunk_id, set()).add(label)

    merged_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)
    merged: list[SearchHit] = []
    for cid in merged_ids[:top_k]:
        hit = by_id[cid]
        source_labels = "+".join(sorted(sources_by_id[cid]))
        merged.append(
            SearchHit(
                chunk_id=hit.chunk_id,
                chunk_index=hit.chunk_index,
                text=hit.text,
                score=scores[cid],
                source=hit.source,
                retrieval_sources=source_labels,
            )
        )
    return merged
