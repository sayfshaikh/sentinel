from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from qdrant_client import QdrantClient

from sentinel.chunking.base import Chunk
from sentinel.eval import label
from sentinel.eval.golden import Answer, Question, find_quote, load
from sentinel.eval.retrieval import bootstrap, evaluate
from sentinel.eval.retrieval_metrics import covers, score
from sentinel.retrieval.dense import DenseRetriever
from sentinel.stores.qdrant_store import QdrantStore

pytestmark = pytest.mark.filterwarnings("ignore:Payload indexes have no effect:UserWarning")

A = Answer("d/1", ((100, 200),))
B = Answer("d/2", ((0, 50),))


def test_covers_needs_half_the_answer() -> None:
    assert covers(("d/1", 150, 400), A)  # holds 50 of 100 characters
    assert not covers(("d/1", 160, 400), A)  # holds 40
    assert not covers(("d/2", 100, 200), A)  # right span, wrong document


def test_scores_each_answer_once() -> None:
    hits = [("d/9", 0, 10), ("d/1", 90, 210), ("d/1", 100, 200), ("d/2", 0, 60)]
    s = score(hits, [A, B])
    assert s.first_hit == (1, 3)
    assert (s.recall(1), s.recall(3), s.recall(5)) == (0.0, 0.5, 1.0)
    assert s.reciprocal_rank == 0.5
    # the second chunk over A (rank 2) earns nothing
    expected = (1 / math.log2(3) + 1 / math.log2(5)) / (1 + 1 / math.log2(3))
    assert s.ndcg() == pytest.approx(expected)


def test_nothing_found_scores_zero() -> None:
    s = score([("d/9", 0, 10)], [A])
    assert (s.recall(10), s.reciprocal_rank, s.ndcg()) == (0.0, 0.0, 0.0)


def test_find_quote_ignores_whitespace() -> None:
    text = (
        "We depend on a\nlimited number of suppliers. "
        "Later: We depend on a limited number of suppliers."
    )
    spans = find_quote(text, "We depend on a   limited number of suppliers.")
    assert len(spans) == 2
    assert text[spans[1][0] : spans[1][1]] == "We depend on a limited number of suppliers."


DOC_ID = "0000320193/0000320193-25-000079"


def _golden(tmp_path: Path, quote: str) -> Path:
    raw = tmp_path / "raw" / "0000320193"
    raw.mkdir(parents=True)
    (raw / "0000320193-25-000079.txt").write_text(
        "Item 1A. Risk Factors\nThe Company depends on component suppliers in Asia.\n"
    )
    path = tmp_path / "golden.json"
    question = {
        "id": "q1",
        "type": "single",
        "question": "Where are Apple's suppliers?",
        "answers": [{"doc_id": DOC_ID, "quote": quote}],
    }
    path.write_text(json.dumps({"version": 1, "questions": [question]}))
    return path


def test_load_resolves_quotes(tmp_path: Path) -> None:
    [q] = load(_golden(tmp_path, "depends on component suppliers in Asia"), tmp_path / "raw")
    assert q.answers[0].spans == ((34, 72),)


def test_load_reports_missing_quote(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="quote not found"):
        load(_golden(tmp_path, "depends on suppliers in Europe only"), tmp_path / "raw")


def test_bootstrap_interval() -> None:
    assert bootstrap([1.0] * 20, [0.0] * 20) == (1.0, 1.0)
    low, high = bootstrap([1.0, 0.0] * 10, [0.0, 1.0] * 10)
    assert low < 0 < high


def test_evaluate_end_to_end() -> None:
    store = QdrantStore(QdrantClient(":memory:"), "t")
    store.create(2)
    chunks = [
        Chunk("d/1", "fixed", 0, "x" * 100, 100, 200),
        Chunk("d/2", "fixed", 0, "y" * 50, 0, 50),
    ]
    store.upsert(chunks=chunks, vectors=[[1.0, 0.0], [0.0, 1.0]])
    questions = [
        Question("q1", "single", "about d/1?", (A,)),
        Question("q2", "single", "d/2?", (B,)),
    ]
    embedder: Any = SimpleNamespace(
        embed_query=lambda q: [1.0, 0.1] if "1" in q else [0.1, 1.0], name="fake", dim=2
    )
    run = evaluate(DenseRetriever(store, embedder), questions)
    assert [s.first_hit for s in run.scores] == [(0,), (0,)]
    assert run.chars_top5 == 150
    assert len(run.latency_ms) == 2


def test_label_writes_a_loadable_golden_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = tmp_path / "raw" / "0000320193"
    raw.mkdir(parents=True)
    (raw / "0000320193-25-000079.txt").write_text(
        "As of September\xa027, 2025 the Company had approximately 166,000 employees.\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(label, "get_settings", lambda: SimpleNamespace(raw_dir=tmp_path / "raw"))
    path = str(tmp_path / "golden.json")

    assert label.main(["--file", path, "question", "s01", "single", "How many employees?"]) == 0
    before = Path(path).read_text(encoding="utf-8")
    quote = "As of September 27, 2025 the Company had approximately 166,000 employees."
    assert (
        label.main(["--file", path, "answer", "s01", DOC_ID, "the Company had 999 employees"]) == 1
    )
    assert label.main(["--file", path, "answer", "s99", DOC_ID, quote]) == 1
    assert Path(path).read_text(encoding="utf-8") == before

    assert label.main(["--file", path, "answer", "s01", DOC_ID, quote]) == 0
    [q] = load(Path(path), tmp_path / "raw")
    assert (q.id, len(q.answers)) == ("s01", 1)
