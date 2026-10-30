from __future__ import annotations

from typing import TYPE_CHECKING

from sentinel.chunking.base import Chunk

if TYPE_CHECKING:
    from sentinel.ingest.corpus import Document


class FixedChunker:
    def __init__(self, size: int = 1000, overlap: int = 200) -> None:
        if overlap >= size:
            raise ValueError(f"overlap {overlap} must be smaller than size {size}")
        self.size = size
        self.overlap = overlap
        self.name = f"fixed-{size}-{overlap}"

    def chunk(self, doc: Document) -> list[Chunk]:
        text = doc.text
        if not text:
            return []

        step = self.size - self.overlap
        chunks: list[Chunk] = []
        for start in range(0, len(text), step):
            end = min(start + self.size, len(text))
            body = text[start:end]
            if body.strip():
                chunks.append(
                    Chunk(
                        doc_id=doc.doc_id,
                        strategy=self.name,
                        index=len(chunks),
                        text=body,
                        start_char=start,
                        end_char=end,
                        meta=doc.meta,
                    )
                )
            if end == len(text):
                break
        return chunks
