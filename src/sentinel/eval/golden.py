"""The golden set: hand-written questions, answers anchored as exact quotes.

An answer is a quote from a filing, not a chunk ID or character offsets. A quote
is readable and checkable by a person, and it resolves to character spans at
load time - which is what lets one golden set score every chunking strategy.

    uv run python -m sentinel.eval.golden      # validate while you label
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from sentinel.config import get_settings

# Single-Fact, Synthesis, Multi-Entity
QuestionType = Literal["single", "synthesis", "multi_entity"]
TARGET: dict[QuestionType, int] = {"single": 20, "synthesis": 20, "multi_entity": 10}
DEFAULT_PATH = get_settings().EVAL_DIR / "golden_set.json"


# --- file schema: a hand-edited file is a trust boundary -------------------


class _Answer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    doc_id: str = Field(pattern=r"^\d{10}/\d{10}-\d{2}-\d{6}$")
    # Short enough that a chunk can hold half of it - see retrieval_metrics.covers.
    quote: str = Field(min_length=20, max_length=500)


class _Question(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    type: QuestionType
    question: str = Field(min_length=10)
    answers: list[_Answer] = Field(min_length=1)


class _GoldenFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: Literal[1]
    questions: list[_Question]


# --- resolved form ------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Answer:
    doc_id: str
    spans: tuple[tuple[int, int], ...]  # every place the quote occurs in the document


@dataclass(frozen=True, slots=True)
class Question:
    id: str
    type: QuestionType
    question: str
    answers: tuple[Answer, ...]


def find_quote(text: str, quote: str) -> list[tuple[int, int]]:
    """Every occurrence of `quote` in `text`, ignoring differences in whitespace."""
    pattern = r"\s+".join(re.escape(word) for word in quote.split())
    return [m.span() for m in re.finditer(pattern, text)]


def load(path: Path, raw_dir: Path) -> list[Question]:
    """Validate the file and resolve every quote. Raises listing every problem."""
    data = _GoldenFile.model_validate_json(path.read_bytes())
    dupes = [qid for qid, n in Counter(q.id for q in data.questions).items() if n > 1]
    problems = [f"duplicate id {qid}" for qid in dupes]

    texts: dict[str, str] = {}
    questions: list[Question] = []
    for q in data.questions:
        answers: list[Answer] = []
        for a in q.answers:
            doc = raw_dir / f"{a.doc_id}.txt"
            if a.doc_id not in texts:
                if not doc.exists():
                    problems.append(f"{q.id}: no such document {doc}")
                    continue
                texts[a.doc_id] = doc.read_text(encoding="utf-8")
            spans = find_quote(texts[a.doc_id], a.quote)
            if not spans:
                problems.append(f"{q.id}: quote not found in {a.doc_id}: {a.quote[:60]!r}")
                continue
            answers.append(Answer(a.doc_id, tuple(spans)))
        questions.append(Question(q.id, q.type, q.question, tuple(answers)))

    if problems:
        raise ValueError("golden set has problems:\n  " + "\n  ".join(problems))
    return questions


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m sentinel.eval.golden")
    parser.add_argument("path", nargs="?", type=Path, default=DEFAULT_PATH)
    args = parser.parse_args(argv)

    try:
        questions = load(args.path, get_settings().raw_dir)
    except (ValueError, OSError) as exc:  # pydantic's ValidationError is a ValueError too
        print(exc, file=sys.stderr)
        return 1

    counts = Counter(q.type for q in questions)
    for kind, target in TARGET.items():
        print(f"{kind:13} {counts[kind]:3} / {target}")
    print(f"{len(questions)} questions, every quote found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
