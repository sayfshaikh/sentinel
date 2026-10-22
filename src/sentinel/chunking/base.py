from __future__ import annotations

import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol

from sentinel.ingest.corpus import Document

# Fixed namespace: chunk IDs must be reproducible across runs and machines.
CHUNK_NAMESPACE = uuid.UUID("e1687cf4-5272-46c3-81cd-803a6ae601a6")


@dataclass(frozen=True, slots=True)
class Chunk:
    doc_id: str
    strategy: str
    index: int
    text: str
    start_char: int
    end_char: int
    meta: Mapping[str, str] = field(default_factory=dict)

    @property
    def chunk_id(self) -> str:
        key = f"{self.strategy}:{self.doc_id}:{self.start_char}:{self.end_char}"
        return str(uuid.uuid5(CHUNK_NAMESPACE, key))

    def overlaps(self, start: int, end: int) -> bool:
        return self.start_char < end and start < self.end_char


class Chunker(Protocol):
    name: str

    def chunk(self, doc: Document) -> list[Chunk]:
        ...

