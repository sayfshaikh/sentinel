"""Cross-encoder reranking: rescore another retriever's candidates."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder

    from sentinel.retrieval.base import Passage, Retriever


class Reranker(Protocol):
    def score(self, query: str, texts: Sequence[str]) -> list[float]: ...


class CrossEncoderReranker:
    """Reads query and passage together, so it is far more accurate than comparing
    two separately-made embeddings - and far slower, so it only sees a shortlist.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        batch_size: int = 16,
        device: str | None = None,
    ) -> None:
        # Imported here, not at module scope: importing torch costs several seconds.
        from sentence_transformers import CrossEncoder

        # device=None lets sentence-transformers pick cuda, then mps (Apple GPU), then cpu.
        self._model: CrossEncoder = CrossEncoder(model_name, device=device)
        self._batch_size = batch_size
        self.name = model_name

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        if not texts:
            return []
        scores = self._model.predict(
            [(query, text) for text in texts],
            batch_size=self._batch_size,
            show_progress_bar=False,
        )
        return [float(s) for s in scores]


class RerankRetriever:
    def __init__(self, inner: Retriever, reranker: Reranker, candidates: int = 20) -> None:
        self.inner = inner
        self.reranker = reranker
        self.candidates = candidates  # the shortlist the cross-encoder reads
        self.name = f"{inner.name}+rerank"

    def retrieve(self, query: str, k: int) -> list[Passage]:
        pool = self.inner.retrieve(query, max(k, self.candidates))
        scores = self.reranker.score(query, [p.text for p in pool])
        ranked = sorted(zip(scores, pool, strict=True), key=lambda pair: pair[0], reverse=True)
        return [dataclasses.replace(p, score=s) for s, p in ranked[:k]]
