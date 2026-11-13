"""BM25 retrieval: sparse term vectors in the same Qdrant collection."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sentinel.retrieval.base import Passage

if TYPE_CHECKING:
    from sentinel.embedding.bm25 import BM25Encoder
    from sentinel.stores.qdrant_store import QdrantStore


class SparseRetriever:
    """Adapter from (store, BM25 encoder) to the Retriever port."""

    def __init__(self, store: QdrantStore, bm25: BM25Encoder) -> None:
        self.store = store
        self.bm25 = bm25
        self.name = "bm25"

    def retrieve(self, query: str, k: int) -> list[Passage]:
        hits = self.store.search_sparse(self.bm25.encode_query(query), k)
        return [Passage.from_payload(h.id, h.score, h.payload) for h in hits]
