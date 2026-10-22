from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np
import pytest

from sentinel.chunking.base import Chunker
from sentinel.chunking.fixed import FixedChunker
from sentinel.chunking.recursive import RecursiveChunker
from sentinel.chunking.semantic import SemanticChunker
from sentinel.ingest.corpus import Document

SIZE, OVERLAP = 400, 80

TEXT = (
    "Item 1A. Risk Factors\n\n"
    "Our business depends on a limited number of suppliers. "
    "A disruption would materially affect resuls.\n\n"
    "We face intense competition in all markets.\n"
) * 20

RUN_ON = "The Company " + "and its subsidiaries " * 150 + "agree.\n\n"


class FakeEmbedder:
    name = "fake"
    dim = 16

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        out = []
        for text in texts:
            digest = hashlib.sha256(text.encode()).digest()[: self.dim]
            v = np.frombuffer(digest, dtype=np.uint8).astype(float) - 127.5
            out.append((v / np.linalg.norm(v)).tolist())
        return out

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]


CHUNKERS: list[Chunker] = [
    FixedChunker(SIZE, OVERLAP),
    RecursiveChunker(SIZE, OVERLAP), 
    SemanticChunker(FakeEmbedder(), min_chars=100, max_chars=SIZE)
]


@pytest.fixture(params=[TEXT, TEXT + RUN_ON + TEXT], ids=["clean", "run-on"])
def doc(request: pytest.FixtureRequest) -> Document:
    return Document(doc_id="0000320193/0000320193-25-000001", text=request.param)

@pytest.mark.parametrize("chunker", CHUNKERS, ids=lambda c: c.name)
def test_spans_are_exact(chunker: Chunker, doc: Document) -> None:
    for chunk in chunker.chunk(doc):
        assert chunk.text == doc.text[chunk.start_char : chunk.end_char]

@pytest.mark.parametrize("chunker", CHUNKERS, ids=lambda c: c.name)
def test_no_chunk_exceeds_size(chunker: Chunker, doc: Document) -> None:
    assert max(len(c.text) for c in chunker.chunk(doc)) <= SIZE 

@pytest.mark.parametrize("chunker", CHUNKERS, ids=lambda c: c.name)
def test_every_word_is_covered(chunker: Chunker, doc: Document) -> None:
    covered = bytearray(len(doc.text))
    for c in chunker.chunk(doc):
        covered[c.start_char : c.end_char] = b"\x01" * (c.end_char - c.start_char)
    missed = [i for i, ch in enumerate(doc.text) if not covered[i] and ch.isalnum()]
    assert not missed, f"uncovered text at {missed[:5]}"

# Used recursive chunking
def test_chunk_ids_are_stable(doc: Document) -> None:
    chunker = RecursiveChunker(SIZE, OVERLAP)

    first = chunker.chunk(doc)
    second = chunker.chunk(doc)
    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]

def test_strategies_produce_different_chunkings(doc: Document) -> None:
    counts = {c.name: len(c.chunk(doc)) for c in CHUNKERS}
    texts = [tuple(ch.text for ch in c.chunk(doc)) for c in CHUNKERS]
    assert len(set(texts)) == len(CHUNKERS), counts