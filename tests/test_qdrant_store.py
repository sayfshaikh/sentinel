from datetime import date

import pytest
from qdrant_client import QdrantClient

from sentinel.chunking.base import Chunk
from sentinel.stores.qdrant_store import QdrantStore

DIM = 4

# In-memory mode ignores payload indexes (it scans instead); the real server uses them.
pytestmark = pytest.mark.filterwarnings("ignore:Payload indexes have no effect:UserWarning")

def _chunk(i: int, form: str, filed: str, ticker: str = "AAPL") -> Chunk:
    return Chunk(
        doc_id=f"d/{i}",
        strategy="test",
        index=i,
        text=f"text {i}",
        start_char=i * 10,
        end_char=i * 10 + 10,
        meta={"ticker": ticker, "form": form, "filing_date": filed, "section": "Item 1A"},
    )


CHUNKS = [
    _chunk(0, "10-K", "2025-02-01"),
    _chunk(1, "8-K", "2025-06-30"),
    _chunk(2, "8-K", "2026-01-15"),
    _chunk(3, "10-Q", "2025-11-05", ticker="JPM"),
]
VECTORS = [[1.0, 0.0, 0.0, 0.0], [0.9, 0.1, 0.0, 0.0], [0.8, 0.2, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]


@pytest.fixture
def store() -> QdrantStore:
    s = QdrantStore(QdrantClient(":memory:"), "test")
    s.create(DIM)
    s.upsert(CHUNKS, VECTORS)
    return s


def test_upsert_is_idempotent(store: QdrantStore) -> None:
    store.upsert(CHUNKS, VECTORS)
    assert store.count() == len(CHUNKS)


def test_payload_keeps_offsets(store: QdrantStore) -> None:
    hit = store.search(VECTORS[1], 1)[0]
    assert hit.id == CHUNKS[1].chunk_id
    span = (hit.payload["doc_id"], hit.payload["start_char"], hit.payload["end_char"])
    assert span == ("d/1", 10, 20)


def test_filter_by_form_and_year(store: QdrantStore) -> None:
    where = store.filter(form="8-K", date_from=date(2025, 1, 1), date_to=date(2026, 1, 1))
    hits = store.search([1.0, 0.0, 0.0, 0.0], 10, where=where)
    assert [h.id for h in hits] == [CHUNKS[1].chunk_id]
    assert store.count(where) == 1


def test_wrong_dimension_is_refused(store: QdrantStore) -> None:
    with pytest.raises(ValueError, match="dimension"):
        store.create(DIM + 1)
