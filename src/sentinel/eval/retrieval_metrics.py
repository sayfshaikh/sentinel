"""Retrieval metrics over character spans: recall@k, MRR, NDCG.

A retrieved chunk *covers* an answer when it holds at least half of the
answer's quote. Scoring is per answer, not per chunk, which keeps one golden
set fair across chunkers: five small chunks of the same paragraph earn credit
once, not five times.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sentinel.eval.golden import Answer

Retrieved = tuple[str, int, int]  # doc_id, start_char, end_char
KS = (1, 3, 5, 10)


def covers(hit: Retrieved, answer: Answer) -> bool:
    doc_id, start, end = hit
    return doc_id == answer.doc_id and any(
        min(end, a_end) - max(start, a_start) >= (a_end - a_start) / 2
        for a_start, a_end in answer.spans
    )


@dataclass(frozen=True, slots=True)
class Score:
    """For each answer, the 0-based rank at which it was first covered, or None."""

    first_hit: tuple[int | None, ...]

    def recall(self, k: int) -> float:
        """Share of the answers covered in the top k."""
        return sum(r is not None and r < k for r in self.first_hit) / len(self.first_hit)

    @property
    def reciprocal_rank(self) -> float:
        found = [r for r in self.first_hit if r is not None]
        return 1 / (min(found) + 1) if found else 0.0

    def ndcg(self, k: int = 10) -> float:
        """Each answer gains once, at the rank where it is first covered."""
        dcg = sum(1 / math.log2(r + 2) for r in self.first_hit if r is not None and r < k)
        ideal = sum(1 / math.log2(r + 2) for r in range(min(len(self.first_hit), k)))
        return dcg / ideal


def score(hits: Sequence[Retrieved], answers: Sequence[Answer]) -> Score:
    return Score(
        tuple(next((r for r, h in enumerate(hits) if covers(h, a)), None) for a in answers)
    )
