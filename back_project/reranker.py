"""Cross-encoder reranking for (query, passage) pairs."""

from __future__ import annotations

from dataclasses import dataclass

from sentence_transformers import CrossEncoder

DEFAULT_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@dataclass(frozen=True)
class RerankResult:
    index: int
    score: float
    text: str


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str = DEFAULT_CROSS_ENCODER_MODEL,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self._model = CrossEncoder(model_name, device=device)

    def rerank(
        self,
        query: str,
        passages: list[str],
        *,
        top_n: int | None = None,
    ) -> list[RerankResult]:
        """
        Score each (query, passage) pair; return sorted by score descending.
        ``top_n`` limits how many results are returned (default: all).
        """
        if not passages:
            return []
        pairs = [[query, p] for p in passages]
        scores = self._model.predict(pairs, show_progress_bar=False)
        scores_list = scores.tolist() if hasattr(scores, "tolist") else list(scores)
        indexed = [
            RerankResult(index=i, score=float(s), text=passages[i])
            for i, s in enumerate(scores_list)
        ]
        indexed.sort(key=lambda r: r.score, reverse=True)
        if top_n is not None:
            indexed = indexed[:top_n]
        return indexed
