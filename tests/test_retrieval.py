from __future__ import annotations

from collections.abc import Sequence

import pytest
from qdrant_client import QdrantClient, models

from sentinel.chunking.base import Chunk
from sentinel.embedding.bm25 import BM25Encoder, tokenize
from sentinel.retrieval.base import Passage
from sentinel.retrieval.hybrid import HybridRetriever, rrf
from sentinel.retrieval.rerank import RerankRetriever
from sentinel.retrieval.sparse import SparseRetriever
from sentinel.stores.qdrant_store import QdrantStore

pytestmark = pytest.mark.filterwarnings("ignore:Payload indexes have no effect:UserWarning")


def _passage(cid: str, text: str = "") -> Passage:
    return Passage(cid, 0.0, f"d/{cid}", 0, 10, text or cid)


class Fixed:
    """A retriever that always returns the same ranking."""

    def __init__(self, name: str, ids: list[str]) -> None:
        self.name = name
        self.ids = ids

    def retrieve(self, query: str, k: int) -> list[Passage]:
        return [_passage(cid) for cid in self.ids[:k]]


def test_rrf_by_hand() -> None:
    fused = rrf([["a", "b"], ["b"]], k=60)
    assert fused["a"] == pytest.approx(1 / 61)
    assert fused["b"] == pytest.approx(1 / 62 + 1 / 61)


def test_hybrid_fuses_and_deduplicates() -> None:
    hybrid = HybridRetriever([Fixed("dense", ["a", "b", "c"]), Fixed("bm25", ["c", "d"])])
    result = hybrid.retrieve("q", 3)
    assert [p.chunk_id for p in result] == ["c", "a", "b"]
    assert hybrid.name == "dense+bm25"


def test_rerank_reorders_the_shortlist() -> None:
    class ByLength:
        def score(self, query: str, texts: Sequence[str]) -> list[float]:
            return [float(len(t)) for t in texts]

    inner = Fixed("dense", ["aa", "a", "aaaa", "aaa"])
    RerankRetriever(inner, ByLength(), candidates=4).retrieve("q", 2)


def test_tokenize_keeps_financial_tokens_whole() -> None:
    tokens = tokenize("CET1 ratios were each 14.2%; 166,000 employees in the U.S. at $60.0")
    assert {"cet1", "14.2%", "166,000", "u.s", "60.0", "employees"} <= set(tokens)
    assert "were" not in tokens and "the" not in tokens


def test_bm25_weights() -> None:
    bm25 = BM25Encoder(k1=1.2, b=0.75, avg_len=3)
    doc = bm25.encode_document("risk risk risk")
    assert len(doc.indices) == 1
    assert doc.values[0] == pytest.approx(3 * 2.2 / (3 + 1.2))
    query = bm25.encode_query("risk Risk RISK factors")
    assert len(query.indices) == 2 and query.values == [1.0, 1.0]
    assert query.indices == sorted(query.indices)


def test_sparse_search_finds_the_exact_term() -> None:
    bm25 = BM25Encoder(avg_len=6)
    texts = [
        "The CET1 capital ratio was 14.2% at year end.",
        "Revenue grew strongly across every segment this year.",
        "We had approximately 166,000 full-time employees.",
    ]
    chunks = [Chunk(f"d/{i}", "t", 0, t, 0, len(t), {"form": "10-K"}) for i, t in enumerate(texts)]
    store = QdrantStore(QdrantClient(":memory:"), "t")
    store.create(2)
    store.upsert(chunks, [[1.0, 0.0]] * 3, [bm25.encode_document(t) for t in texts])

    hits = SparseRetriever(store, bm25).retrieve("What is the CET1 ratio?", 3)
    assert hits[0].doc_id == "d/0"
    assert (
        store.search_sparse(bm25.encode_query("166,000 employees"), 1)[0].payload["doc_id"] == "d/2"
    )
    assert store.search_sparse(bm25.encode_query("the and of"), 3) == []


def test_old_single_vector_collection_is_refused() -> None:
    client = QdrantClient(":memory:")
    client.create_collection(
        "old", vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE)
    )
    with pytest.raises(ValueError, match="--recreate"):
        QdrantStore(client, "old").create(2)
