"""Dense retrieval: embed the query, nearest neighbours in Qdrant."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sentinel.embedding.base import Embedder
    from sentinel.stores.qdrant_store import QdrantStore
from sentinel.retrieval.base import Passage


class DenseRetriever:
    """Adapter from (store, embedder) to the Retriever port."""

    def __init__(self, store: QdrantStore, embedder: Embedder) -> None:
        self.store = store
        self.embedder = embedder
        self.name = "dense"

    def retrieve(self, query: str, k: int) -> list[Passage]:
        hits = self.store.search(self.embedder.embed_query(query), k)
        return [Passage.from_payload(h.id, h.score, h.payload) for h in hits]
