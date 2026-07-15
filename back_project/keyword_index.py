"""In-memory BM25 keyword index kept in sync with vector store chunk IDs."""

from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

from vector_store import SearchHit

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class KeywordIndex:
    def __init__(self) -> None:
        self._ids: list[str] = []
        self._texts: list[str] = []
        self._chunk_indices: list[int] = []
        self._sources: list[str | None] = []
        self._bm25: BM25Okapi | None = None

    def add_chunks(
        self,
        chunk_ids: list[str],
        chunk_texts: list[str],
        *,
        source: str | None = None,
        start_chunk_index: int = 0,
    ) -> None:
        for offset, (chunk_id, text) in enumerate(
            zip(chunk_ids, chunk_texts, strict=True)
        ):
            self._ids.append(chunk_id)
            self._texts.append(text)
            self._chunk_indices.append(start_chunk_index + offset)
            self._sources.append(source)
        self._rebuild()

    def _rebuild(self) -> None:
        if not self._texts:
            self._bm25 = None
            return
        corpus = [tokenize(text) for text in self._texts]
        self._bm25 = BM25Okapi(corpus)

    def search(self, query: str, *, top_k: int) -> list[SearchHit]:
        if top_k <= 0 or not self._bm25:
            return []

        tokens = tokenize(query)
        if not tokens:
            return []

        scores = self._bm25.get_scores(tokens)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        hits: list[SearchHit] = []
        for i in ranked[:top_k]:
            score = float(scores[i])
            if score <= 0:
                continue
            hits.append(
                SearchHit(
                    chunk_id=self._ids[i],
                    chunk_index=self._chunk_indices[i],
                    text=self._texts[i],
                    score=score,
                    source=self._sources[i],
                    retrieval_sources="keyword",
                )
            )
        return hits

    def has_document(self) -> bool:
        return bool(self._ids)
