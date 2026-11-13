"""BM25 sparse vectors, written by hand.

Qdrant computes IDF server-side (the collection's sparse vector uses
Modifier.IDF), so a document vector carries only BM25's term-frequency part:

    tf * (k1 + 1) / (tf + k1 * (1 - b + b * length / avg_len))

and a query vector is 1.0 per distinct term. Qdrant then scores a chunk as the
sum over shared terms of idf(term) * document weight - which is BM25.
"""

from __future__ import annotations

import re
import zlib
from collections import Counter
from dataclasses import dataclass

# Numbers keep their punctuation, so the exact figures dense search misses stay
# whole: "14.2%", "166,000", "60.0". Words keep internal dots: "u.s", "cet1".
_TOKEN = re.compile(r"\d+(?:[.,]\d+)*%?|[a-z0-9]+(?:\.[a-z0-9]+)*")

# shortcut: a short stopword list and no stemming ("supplier" != "suppliers");
# add a stemmer if Day 5's per-question results show plural misses.
_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "been", "by", "for", "from", "has", "have", "in",
    "is", "it", "its", "of", "on", "or", "s", "that", "the", "their", "this", "to", "was", "we",
    "were", "which", "will", "with", "our", "us"
})  # fmt: skip


@dataclass(frozen=True, slots=True)
class SparseVec:
    indices: list[int]
    values: list[float]


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS]


def _index(term: str) -> int:
    # crc32 fits Qdrant's uint32 sparse indices; ~1 collision expected per 100k terms.
    return zlib.crc32(term.encode("utf-8"))


def _sparse(weights: dict[int, float]) -> SparseVec:
    indices = sorted(weights)
    return SparseVec(indices, [weights[i] for i in indices])


class BM25Encoder:
    def __init__(self, k1: float = 1.2, b: float = 0.75, avg_len: float = 160.0) -> None:
        self.k1 = k1
        self.b = b
        # shortcut: a fixed average length, as Qdrant's own BM25 uses; the true
        # corpus average would need a second pass over every chunk.
        self.avg_len = avg_len

    def encode_document(self, text: str) -> SparseVec:
        terms = tokenize(text)
        norm = self.k1 * (1 - self.b + self.b * len(terms) / self.avg_len)
        weights: dict[int, float] = {}
        for term, tf in Counter(terms).items():
            i = _index(term)  # summing keeps indices unique if two terms collide
            weights[i] = weights.get(i, 0.0) + tf * (self.k1 + 1) / (tf + norm)
        return _sparse(weights)

    def encode_query(self, text: str) -> SparseVec:
        return _sparse(dict.fromkeys((_index(t) for t in tokenize(text)), 1.0))
