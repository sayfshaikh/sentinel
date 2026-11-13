"""Score every chunking strategy against the golden set.

    uv run python -m sentinel.eval.retrieval

Needs one Qdrant collection per strategy, named <collection>_<strategy>
(filings_fixed, filings_recursive, filings_semantic). Prints tables with
tabulate and writes per-question results to eval/results/ so every number is reproducible.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

from tabulate import tabulate

from sentinel.config import embedder_from_config, get_settings
from sentinel.eval.golden import DEFAULT_PATH, TARGET, Question, load
from sentinel.eval.retrieval_metrics import KS, Score, score
from sentinel.retrieval.dense import DenseRetriever
from sentinel.stores.qdrant_store import QdrantStore

if TYPE_CHECKING:
    from sentinel.retrieval.base import Retriever

STRATEGIES = ("fixed", "recursive", "semantic")


@dataclass(frozen=True, slots=True)
class Run:
    scores: list[Score]
    chars_top5: float
    latency_ms: list[float]


def evaluate(
    retriever: Retriever,
    questions: Sequence[Question],
    k: int = 10,
) -> Run:
    scores: list[Score] = []
    context: list[int] = []
    latency: list[float] = []
    for question in questions:
        started = time.perf_counter()
        passages = retriever.retrieve(question.question, k)
        latency.append((time.perf_counter() - started) * 1000)
        hits = [(p.doc_id, p.start_char, p.end_char) for p in passages]
        scores.append(score(hits, question.answers))
        context.append(sum(end - start for _, start, end in hits[:5]))
    return Run(scores, statistics.fmean(context), latency)


def bootstrap(a: Sequence[float], b: Sequence[float], n: int = 2000) -> tuple[float, float]:
    """95% interval for mean(a) - mean(b), resampling questions as pairs."""
    rng = random.Random(0)
    deltas = [x - y for x, y in zip(a, b, strict=True)]
    means = sorted(statistics.fmean(rng.choices(deltas, k=len(deltas))) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n) - 1]


def by_type_table(questions: Sequence[Question], runs: Mapping[str, Run], label: str) -> str:
    kinds = [kind for kind in TARGET if any(q.type == kind for q in questions)]
    sizes = {kind: sum(q.type == kind for q in questions) for kind in kinds}
    rows = [
        [
            name,
            *(
                statistics.fmean(
                    s.recall(5)
                    for q, s in zip(questions, run.scores, strict=True)
                    if q.type == kind
                )
                for kind in kinds
            ),
        ]
        for name, run in runs.items()
    ]
    return tabulate(
        rows,
        ["Chunker", *(f"{kind} (n={sizes[kind]})" for kind in kinds)],
        tablefmt="github",
        floatfmt=".3f",
    )


def save(path: Path, questions: Sequence[Question], runs: Mapping[str, Run], **info: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ranks = {
        name: {q.id: list(s.first_hit) for q, s in zip(questions, run.scores, strict=True)}
        for name, run in runs.items()
    }
    path.write_text(json.dumps({**info, "first_hit_rank": ranks}, indent=2), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m sentinel.eval.retrieval")
    parser.add_argument("--golden", type=Path, default=DEFAULT_PATH)
    parser.add_argument("--strategies", nargs="+", default=list(STRATEGIES))
    args = parser.parse_args(argv)

    settings = get_settings()
    questions = load(args.golden, settings.raw_dir)
    stores = {
        s: QdrantStore.connect(settings.qdrant_url, f"{settings.qdrant_collection}_{s}")
        for s in args.strategies
    }
    if missing := [
        st.collection for st in stores.values() if not st.client.collection_exists(st.collection)
    ]:
        print(f"missing collections {missing} - index them first")
        return 1

    # One query embedding per question, shared by every strategy: the chunker is
    # the only thing that varies between rows.
    embedder = embedder_from_config(settings)
    runs = {s: evaluate(DenseRetriever(store, embedder), questions) for s, store in stores.items()}

    rows = [
        [
            strategy,
            *(statistics.fmean(s.recall(k) for s in run.scores) for k in KS),
            statistics.fmean(s.reciprocal_rank for s in run.scores),
            statistics.fmean(s.ndcg() for s in run.scores),
            run.chars_top5,
        ]
        for strategy, run in runs.items()
    ]

    headers = ["Chunker", *(f"recall@{k}" for k in KS), "MRR", "NDCG@10", "chars in top 5"]
    floatfmt = [".3f"] * (len(headers) - 1) + [",.0f"]
    print(f"\n## Chunking comparison — {len(questions)} questions, `{embedder.name}`\n")
    print(tabulate(rows, headers, tablefmt="github", floatfmt=floatfmt))
    print("\n### recall@5 by question type\n")
    print(by_type_table(questions, runs, "Chunker"))

    by_recall = sorted(runs, key=lambda s: statistics.fmean(x.recall(5) for x in runs[s].scores))
    if len(by_recall) > 1:
        best, second = by_recall[-1], by_recall[-2]
        a = [x.recall(5) for x in runs[best].scores]
        b = [x.recall(5) for x in runs[second].scores]
        low, high = bootstrap(a, b)
        verdict = (
            "a real difference"
            if low > 0
            else "within noise at this sample size - prefer the cheaper chunker"
        )
        print(
            f"\n**{best}** vs {second}: recall@5 {statistics.fmean(a) - statistics.fmean(b):+.3f}, "
            f"95% CI [{low:+.3f}, {high:+.3f}] — {verdict}."
        )

    out = get_settings().EVAL_DIR / "results" / f"{date.today()}-chunking.json"
    save(out, questions, runs, embedding_model=embedder.name, golden_set=str(args.golden))
    print(f"\nPer-question results: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
