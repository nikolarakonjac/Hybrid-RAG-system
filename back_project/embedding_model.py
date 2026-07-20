from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer

FloatArray = NDArray[np.floating]

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class EmbeddingModel:
    """
    Sentence embeddings via a transformer model. Encodings are L2-normalized by
    default so cosine similarity equals the dot product of embedding rows.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self._use_bge_query_instruction = model_name.startswith("BAAI/bge-")
        self._model = SentenceTransformer(model_name, device=device)
        dim = self._model.get_sentence_embedding_dimension()
        if dim is None:
            raise RuntimeError("Model did not report an embedding dimension")
        self._embedding_dim: int = dim

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def encode(
        self,
        sentences: Sequence[str],
        *,
        batch_size: int = 32,
        normalize: bool = True,
    ) -> FloatArray:
        """
        Embed a batch of sentences. Empty input returns shape (0, embedding_dim).
        """
        if not sentences:
            return np.zeros((0, self._embedding_dim), dtype=np.float32)
        out = self._model.encode(
            list(sentences),
            batch_size=batch_size,
            normalize_embeddings=normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return out.astype(np.float32, copy=False)

    def _prepare_query(self, query: str) -> str:
        if self._use_bge_query_instruction:
            return f"{BGE_QUERY_INSTRUCTION}{query}"
        return query

    def encode_one(
        self,
        sentence: str,
        *,
        batch_size: int = 32,
        normalize: bool = True,
        for_query: bool = True,
    ) -> FloatArray:
        """Single sentence; returns shape (embedding_dim,)."""
        text = self._prepare_query(sentence) if for_query else sentence
        emb = self.encode([text], batch_size=batch_size, normalize=normalize)
        return emb[0]

    def cosine_similarity(self, a: FloatArray, b: FloatArray) -> float | FloatArray:
        """
        Cosine similarity between row vectors of ``a`` and ``b``.
        For L2-normalized embeddings this is ``a @ b.T``.

        - Both 1D same length: scalar float.
        - ``a`` (n, d), ``b`` (m, d): (n, m) matrix.
        """
        a = np.asarray(a, dtype=np.float32)
        b = np.asarray(b, dtype=np.float32)
        if a.ndim == 1 and b.ndim == 1:
            if a.shape[0] != b.shape[0]:
                raise ValueError("1D vectors must have the same length")
            return float(np.dot(a, b))
        a2 = a if a.ndim == 2 else np.expand_dims(a, 0)
        b2 = b if b.ndim == 2 else np.expand_dims(b, 0)
        sims: FloatArray = a2 @ b2.T
        if sims.shape == (1, 1):
            return float(sims[0, 0])
        if sims.shape[0] == 1:
            return sims[0]
        if sims.shape[1] == 1:
            return sims[:, 0]
        return sims

    def pairwise_similarity_matrix(self, embeddings: FloatArray) -> FloatArray:
        """
        All pairwise cosine similarities between rows of ``embeddings``.
        shape (n, n). For normalized rows, ``embeddings @ embeddings.T``.
        """
        e = np.asarray(embeddings, dtype=np.float32)
        if e.ndim != 2:
            raise ValueError("embeddings must be 2D (n, dim)")
        return e @ e.T

    def rank_by_similarity(
        self,
        query: str,
        candidates: Sequence[str],
        *,
        top_k: int | None = None,
        normalize: bool = True,
    ) -> list[tuple[int, float, str]]:
        """
        Embed ``query`` and each candidate, return sorted (index, score, text).
        Scores are cosine similarities in [~0, 1] for normalized vectors.
        """
        if not candidates:
            return []
        q = self.encode_one(query, normalize=normalize)
        cand_emb = self.encode(list(candidates), normalize=normalize)
        scores = (cand_emb @ q).astype(np.float64)
        order = np.argsort(-scores)
        k = len(order) if top_k is None else min(top_k, len(order))
        cand_list = list(candidates)
        return [(int(i), float(scores[i]), cand_list[i]) for i in order[:k]]
