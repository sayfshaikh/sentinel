from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING, Any, cast

from qdrant_client import QdrantClient, models

if TYPE_CHECKING:
    from sentinel.chunking.base import Chunk
    from sentinel.embedding.bm25 import SparseVec

DENSE = "dense"
SPARSE = "bm25"
_KEYWORD_FIELDS = ("ticker", "form", "section")
_UPSERT_BATCH = 256  # keeps each request far below Qdrant's 32 MB request limit


@dataclass(frozen=True, slots=True)
class Point:
    id: str
    vector: list[float]
    payload: dict[str, Any]
    sparse: SparseVec | None = None


@dataclass(frozen=True, slots=True)
class Hit:
    id: str
    score: float
    payload: dict[str, Any]


class QdrantStore:
    def __init__(self, client: QdrantClient, collection: str) -> None:
        self.client = client
        self.collection = collection

    @classmethod
    def connect(cls, url: str, collection: str) -> QdrantStore:
        return cls(QdrantClient(url=url, timeout=120), collection)

    def create(
        self, dim: int, *, recreate: bool = False, on_disk: bool = False, quantize: bool = False
    ) -> None:
        if self.client.collection_exists(self.collection):
            if not recreate:
                self._check_dim(dim)
                return
            self.client.delete_collection(self.collection)

        self.client.create_collection(
            collection_name=self.collection,
            vectors_config={
                DENSE: models.VectorParams(
                    size=dim, distance=models.Distance.COSINE, on_disk=on_disk
                )
            },
            sparse_vectors_config={SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)},
            hnsw_config=models.HnswConfigDiff(m=0),
            optimizers_config=models.OptimizersConfigDiff(indexing_threshold=1),
            quantization_config=(
                models.ScalarQuantization(
                    scalar=models.ScalarQuantizationConfig(
                        type=models.ScalarType.INT8, quantile=0.99, always_ram=True
                    )
                )
                if quantize
                else None
            ),
        )
        for field in _KEYWORD_FIELDS:
            self._index(field, models.PayloadSchemaType.KEYWORD)
        self._index("filing_date", models.PayloadSchemaType.DATETIME)
        self._index("chunk_index", models.PayloadSchemaType.INTEGER)

    def build_index(self, m: int = 16, ef_construct: int = 100, timeout_s: float = 1800) -> float:
        """
        Switch HNSW on and block until it is built. Returns seconds taken.
        """
        started = time.perf_counter()
        self.client.update_collection(
            collection_name=self.collection,
            hnsw_config=models.HnswConfigDiff(m=m, ef_construct=ef_construct),
        )
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            info = self.client.get_collection(self.collection)
            if info.status == models.CollectionStatus.GREY:
                self.client.update_collection(
                    collection_name=self.collection,
                    optimizers_config=models.OptimizersConfigDiff(),
                )
            elif info.status == models.CollectionStatus.GREEN and (
                info.indexed_vectors_count or 0
            ) >= (info.points_count or 0):
                return time.perf_counter() - started
            time.sleep(0.25)
        raise TimeoutError(f"{self.collection}: HNSW not built after {timeout_s:.0f}s")

    def upsert(
        self,
        chunks: Sequence[Chunk],
        vectors: Sequence[Sequence[float]],
        sparse: Sequence[SparseVec] | None = None,
    ) -> None:
        sparse_vectors = sparse if sparse is not None else [None] * len(chunks)
        self.upsert_points(
            [
                Point(
                    id=chunk.chunk_id,
                    vector=list(vector),
                    payload={
                        **chunk.meta,
                        "doc_id": chunk.doc_id,
                        "strategy": chunk.strategy,
                        "chunk_index": chunk.index,
                        "start_char": chunk.start_char,
                        "end_char": chunk.end_char,
                        "text": chunk.text,
                    },
                    sparse=sparse_vec,
                )
                for chunk, vector, sparse_vec in zip(chunks, vectors, sparse_vectors, strict=True)
            ]
        )

    def upsert_points(self, points: Sequence[Point]) -> None:
        for i in range(0, len(points), _UPSERT_BATCH):
            self.client.upsert(
                collection_name=self.collection,
                points=[
                    models.PointStruct(id=p.id, vector=self._vectors(p), payload=p.payload)
                    for p in points[i : i + _UPSERT_BATCH]
                ],
                wait=True,
            )

    @staticmethod
    def _vectors(point: Point) -> dict[str, Any]:
        vectors: dict[str, Any] = {DENSE: point.vector}
        if point.sparse and point.sparse.indices:  # a chunk of only stopwords has no terms
            vectors[SPARSE] = models.SparseVector(
                indices=point.sparse.indices, values=point.sparse.values
            )
        return vectors

    def drop(self) -> None:
        self.client.delete_collection(self.collection)

    def count(self, where: models.Filter | None = None) -> int:
        return self.client.count(
            collection_name=self.collection, count_filter=where, exact=True
        ).count

    def search(
        self,
        vector: Sequence[float],
        limit: int = 20,
        *,
        where: models.Filter | None = None,
        hnsw_ef: int | None = None,
        exact: bool = False,
        rescore: bool | None = None,
    ) -> list[Hit]:
        """Nearest neighbours, `rescore` only matters on a quantised collection."""
        quantization = (
            None
            if rescore is None
            else models.QuantizationSearchParams(
                rescore=rescore, oversampling=2.0 if rescore else None
            )
        )
        response = self.client.query_points(
            collection_name=self.collection,
            query=list(vector),
            using=DENSE,
            query_filter=where,
            limit=limit,
            with_payload=True,
            search_params=models.SearchParams(
                hnsw_ef=hnsw_ef, exact=exact, quantization=quantization
            ),
        )
        return [Hit(str(p.id), p.score, p.payload or {}) for p in response.points]

    def search_sparse(
        self,
        query: SparseVec,
        limit: int = 10,
        *,
        where: models.Filter | None = None,
    ) -> list[Hit]:
        """BM25 search: Qdrant multiplies each query term by its IDF."""
        if not query.indices:  # a query of only stopwords matches nothing
            return []
        response = self.client.query_points(
            collection_name=self.collection,
            query=models.SparseVector(indices=query.indices, values=query.values),
            using=SPARSE,
            query_filter=where,
            limit=limit,
            with_payload=True,
        )
        return [Hit(str(p.id), p.score, p.payload or {}) for p in response.points]

    def all_points(self) -> list[Point]:
        out: list[Point] = []
        offset: Any = None
        while True:
            records, offset = self.client.scroll(
                collection_name=self.collection,
                limit=1000,
                offset=offset,
                with_payload=True,
                with_vectors=[DENSE],
            )

            out += [
                Point(str(r.id), cast("dict[str, list[float]]", r.vector)[DENSE], r.payload or {})
                for r in records
            ]
            if offset is None:
                return out

    @staticmethod
    def filter(
        *,
        form: str | None = None,
        ticker: str | None = None,
        section: str | None = None,
        date_from: date | None = None,
        date_to: date | None = None,
    ) -> models.Filter:
        must: list[models.Condition] = [
            models.FieldCondition(key=key, match=models.MatchValue(value=value))
            for key, value in (("form", form), ("ticker", ticker), ("section", section))
            if value is not None
        ]
        if date_from or date_to:
            must.append(
                models.FieldCondition(
                    key="filing_date", range=models.DatetimeRange(gte=date_from, lt=date_to)
                )
            )
        return models.Filter(must=must)

    def _index(self, field: str, schema: models.PayloadSchemaType) -> None:
        self.client.create_payload_index(
            collection_name=self.collection, field_name=field, field_schema=schema, wait=True
        )

    def _check_dim(self, dim: int) -> None:
        vectors = self.client.get_collection(self.collection).config.params.vectors
        dense = vectors.get(DENSE) if isinstance(vectors, dict) else None
        if dense is None:
            raise ValueError(
                f"collection {self.collection!r} was created without a dense vector field; "
                "re-run with --recreate"
            )
        if dense.size != dim:
            raise ValueError(
                f"collection {self.collection!r} has dimension {dense.size} but the embedder "
                f"produces {dim}; re-run with --recreate after changing the embedding model"
            )
