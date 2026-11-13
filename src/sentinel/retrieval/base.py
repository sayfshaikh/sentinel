from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class Passage:
    chunk_id: str
    score: float
    doc_id: str
    start_char: int
    end_char: int
    text: str
    meta: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, chunk_id: str, score: float, payload: Mapping[str, Any]) -> Passage:
        """Build from a stored chunk's payload (see QdrantStore.upsert)."""
        return cls(
            chunk_id=chunk_id,
            score=score,
            doc_id=str(payload["doc_id"]),
            start_char=int(payload["start_char"]),
            end_char=int(payload["end_char"]),
            text=str(payload.get("text", "")),
            meta=payload,
        )


class Retriever(Protocol):
    name: str

    def retrieve(self, query: str, k: int) -> list[Passage]:
        """The k best passages for a query, best first."""
        ...
