"""In-memory vector store via ChromaDB (ephemeral client; no disk persistence)."""

from __future__ import annotations

import uuid
from dataclasses import dataclass

import chromadb
import numpy as np

FloatArray = np.ndarray


@dataclass(frozen=True)
class SearchHit:
    chunk_id: str
    chunk_index: int
    text: str
    score: float
    source: str | None
    retrieval_sources: str | None = None


class ChromaVectorStore:
    """
    Chroma collection with cosine space on bi-encoder embeddings (L2-normalized).
    ``add_chunks`` appends; multiple PDFs can be ingested over time.
    """

    COLLECTION_NAME = "rag_chunks"
    # Chroma enforces a max records-per-add limit (varies by version; ~5461 observed).
    ADD_BATCH_SIZE = 5000

    def __init__(self) -> None:
        self._client = chromadb.EphemeralClient()
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(
        self,
        chunk_texts: list[str],
        embeddings: FloatArray,
        *,
        source: str | None = None,
    ) -> list[str]:
        """Append chunks and embeddings for one ingest batch. Returns chunk IDs."""
        emb = np.asarray(embeddings, dtype=np.float32)
        if len(chunk_texts) == 0:
            if emb.size != 0:
                raise ValueError("embeddings must be empty when there are no chunks")
            return []
        if emb.ndim != 2:
            raise ValueError("embeddings must be 2D (n_chunks, dim)")
        if emb.shape[0] != len(chunk_texts):
            raise ValueError("embeddings row count must match chunk_texts length")

        ids = [str(uuid.uuid4()) for _ in chunk_texts]
        metadatas = [
            {"source": source or "", "chunk_index": int(i)}
            for i in range(len(chunk_texts))
        ]
        n = len(chunk_texts)
        for start in range(0, n, self.ADD_BATCH_SIZE):
            end = min(start + self.ADD_BATCH_SIZE, n)
            self._collection.add(
                ids=ids[start:end],
                embeddings=emb[start:end].tolist(),
                documents=chunk_texts[start:end],
                metadatas=metadatas[start:end],
            )
        return ids

    def search(
        self,
        query_embedding: FloatArray,
        *,
        top_k: int,
    ) -> list[SearchHit]:
        """Top ``top_k`` chunks by cosine similarity (Chroma cosine distance → score)."""
        if top_k <= 0:
            return []
        n = self._collection.count()
        if n == 0:
            return []

        q = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
        k = min(top_k, n)
        res = self._collection.query(
            query_embeddings=[q.tolist()],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        ids_list = res["ids"][0]
        docs_list = res["documents"][0]
        meta_list = res["metadatas"][0]
        dist_list = res["distances"][0]

        hits: list[SearchHit] = []
        for cid, text, meta, dist in zip(
            ids_list, docs_list, meta_list, dist_list, strict=True
        ):
            src = meta.get("source") if meta else None
            if src == "":
                src = None
            chunk_index = int(meta.get("chunk_index", 0)) if meta else 0
            # Chroma cosine distance = 1 - cosine_similarity for unit vectors
            score = 1.0 - float(dist)
            hits.append(
                SearchHit(
                    chunk_id=cid,
                    chunk_index=chunk_index,
                    text=text or "",
                    score=score,
                    source=src,
                    retrieval_sources="vector",
                )
            )
        return hits

    def has_document(self) -> bool:
        return self._collection.count() > 0
