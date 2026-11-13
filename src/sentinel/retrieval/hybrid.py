"""Hybrid retrieval: reciprocal rank fusion over any set of retrievers."""

from __future__ import annotations

import dataclasses
from collections import defaultdict
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sentinel.retrieval.base import Passage, Retriever


def rrf(rankings: Sequence[Sequence[str]], k: int = 60) -> dict[str, float]:
    """Reciprocal rank fusion: each ranking adds 1 / (k + rank) to every item, rank from 1.

    Only ranks are used, never raw scores, so a cosine similarity and a BM25
    score can be fused without normalising either. k = 60 is the value from the
    original paper (Cormack et al., 2009); it damps how much the very top rank
    dominates.
    """
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] += 1.0 / (k + rank)
    return dict(scores)


class HybridRetriever:
    def __init__(
        self, retrievers: Sequence[Retriever], candidates: int = 50, k_rrf: int = 60
    ) -> None:
        self.retrievers = list(retrievers)
        self.candidates = candidates  # how deep each retriever looks before fusing
        self.k_rrf = k_rrf
        self.name = "+".join(r.name for r in self.retrievers)

    def retrieve(self, query: str, k: int) -> list[Passage]:
        results = [r.retrieve(query, max(k, self.candidates)) for r in self.retrievers]
        passages = {p.chunk_id: p for result in results for p in result}
        fused = rrf([[p.chunk_id for p in result] for result in results], self.k_rrf)
        best = sorted(fused, key=fused.__getitem__, reverse=True)[:k]
        return [dataclasses.replace(passages[cid], score=fused[cid]) for cid in best]
