from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from sentinel.chunking.base import Chunk

if TYPE_CHECKING:
    from sentinel.ingest.corpus import Document

# Coarsest first. Fall through only when a piece is still too large.
DEFAULT_SEPARATORS: tuple[str, ...] = ("\n\n", "\n", ". ", " ", "")

Span = tuple[int, int]


class RecursiveChunker:
    def __init__(
        self, size: int = 1000, overlap: int = 200, separators: Sequence[str] = DEFAULT_SEPARATORS
    ) -> None:
        if overlap >= size:
            raise ValueError(f"overlap {overlap} must be smaller than size {size}")
        self.size = size
        self.overlap = overlap
        self.separators = tuple(separators)
        self.name = f"recursive-{size}-{overlap}"

    def chunk(self, doc: Document) -> list[Chunk]:
        text = doc.text
        if not text:
            return []

        atoms = self._split(text, 0, self.separators)
        spans = self._merge(atoms)
        return [
            Chunk(
                doc_id=doc.doc_id,
                strategy=self.name,
                index=i,
                text=text[start:end],
                start_char=start,
                end_char=end,
                meta=doc.meta,
            )
            for i, (start, end) in enumerate(spans)
            if text[start:end].strip()
        ]

    def _split(self, text: str, offset: int, seps: Sequence[str]) -> list[Span]:
        if len(text) <= self.size:
            return [(offset, offset + len(text))] if text else []

        if not seps:
            return self._hard_cut(text, offset)

        sep, rest = seps[0], seps[1:]
        if sep == "":
            return self._hard_cut(text, offset)

        pieces: list[Span] = []
        cursor = 0
        for piece in text.split(sep):
            start, end = cursor, cursor + len(piece)
            if piece:
                if len(piece) > self.size:
                    pieces.extend(self._split(piece, offset + start, rest))
                else:
                    pieces.append((offset + start, offset + end))
            cursor = end + len(sep)
        return pieces

    def _hard_cut(self, text: str, offset: int) -> list[Span]:
        return [
            (offset + i, offset + min(i + self.size, len(text)))
            for i in range(0, len(text), self.size)
        ]

    def _merge(self, atoms: list[Span]) -> list[Span]:
        if not atoms:
            return []

        out: list[Span] = []
        cur_start, cur_end = atoms[0]
        for _, end in atoms[1:]:
            if end - cur_start <= self.size:
                cur_end = end
            else:
                out.append((cur_start, cur_end))
                cur_start = max(0, cur_end - self.overlap, end - self.size)
                cur_end = end
        out.append((cur_start, cur_end))
        return out
