from __future__ import annotations

import re

import numpy as np

from sentinel.chunking.base import Chunk
from sentinel.ingest.corpus import Document
from sentinel.embedding.base import Embedder


# Sentence splitting regex: matches a sequence of non-sentence-ending characters 
# followed by optional sentence-ending punctuation.
_SENTENCE_RE = re.compile(r"[^\n.!?]+[.!?]*")


class SemanticChunker:
    def __init__(
        self,
        embedder: Embedder,
        percentile: float = 0.5,
        min_chars: int = 200, 
        max_chars: int = 2000,
    ) -> None:
        self.embedder = embedder
        self.percentile = percentile
        self.min_chars = min_chars
        self.max_chars = max_chars
        self.name = f"semantic-p{int(percentile)}"

    def chunk(self, doc: Document) -> list[Chunk]:
        spans = self._sentences(doc.text)
        if len(spans) < 2:
            return self._single(doc, spans)

        vectors = np.asarray(self.embedder.embed([doc.text[s:e] for s, e in spans]))
        similarity = np.sum(vectors[:-1] * vectors[1:], axis=1)
        distance = 1.0 - similarity
        threshold = float(np.percentile(distance, self.percentile))

        chunks: list[Chunk] = []
        start = spans[0][0]
        for i, (_, end) in enumerate(spans[:-1]):
            topic_shift = end - start >= self.min_chars and distance[i] >= threshold

            would_overflow = spans[i + 1][1] - start > self.max_chars
            if topic_shift or would_overflow:
                chunks.append(self._make(doc, len(chunks), start, end))
                start = spans[i + 1][0]

        chunks.append(self._make(doc, len(chunks), start, spans[-1][1]))
        return [c for c in chunks if c.text.strip()]

    def _sentences(self, text: str) -> list[tuple[int, int]]:
        spans: list[tuple[int, int]] = []
        for match in _SENTENCE_RE.finditer(text):
            if not match.group().strip():
                continue
            start, end = match.span()
            for cut in range(start, end, self.max_chars):
                spans.append((cut, min(cut + self.max_chars, end)))
        return spans

    def _single(self, doc: Document, spans: list[tuple[int, int]]) -> list[Chunk]:
        if not spans:
            return []
        return [self._make(doc, 0, spans[0][0], spans[-1][1])]

    def _make(self, doc: Document, index: int, start: int, end: int) -> Chunk:
        return Chunk(
            doc_id=doc.doc_id,
            strategy=self.name,
            index=index,
            text=doc.text[start:end],
            start_char=start,
            end_char=end,
            meta=doc.meta,
        )