"""Score each retrieval stage against the golden set.

    uv run python -m sentinel.eval.pipeline

Dense, then dense + BM25 fused with RRF, then that hybrid reranked by a
cross-encoder - all on one collection, so the stage is the only thing that
varies. Each step's change is reported with 95% bootstrap interval.
"""

from __future__ import annotations

import argparse
import itertools
import statistics
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

from tabulate import tabulate

from sentinel.config import Stage, embedder_from_config, get_settings, retriever_from_config
from sentinel.eval.golden import DEFAULT_PATH, load
from sentinel.eval.retrieval import Run, bootstrap, by_type_table, evaluate, save
from sentinel.stores.qdrant_store import QdrantStore

if TYPE_CHECKING:
    from sentinel.eval.retrieval_metrics import Score

STAGES: tuple[Stage, ...] = ("dense", "hybrid", "rerank")
LABELS = {"dense": "Dense", "hybrid": "+ BM25 (RRF)", "rerank": "+ Cross-Encoder rerank"}


def _pct(values: Sequence[float], q: int) -> float:
    return statistics.quantiles(values, n=100)[q - 1] if len(values) > 1 else values[0]


def _mrr(score: Score) -> float:
    return score.reciprocal_rank


def _recall5(score: Score) -> float:
    return score.recall(5)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m sentinel.eval.pipeline")
    parser.add_argument(
        "--golden",
        type=Path,
        default=DEFAULT_PATH,
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=STAGES,
        default=list(STAGES),
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    questions = load(args.golden, settings.raw_dir)
    store = QdrantStore.connect(settings.qdrant_url, settings.qdrant_collection)
    if not store.client.collection_exists(store.collection):
        print(f"missing collection {store.collection} - index it first")
        return 1

    embedder = embedder_from_config(settings)
    runs: dict[str, Run] = {}
    for stage in args.stages:
        runs[LABELS[stage]] = evaluate(retriever_from_config(stage, settings, embedder), questions)

    rows = [
        [
            name,
            statistics.fmean(s.recall(1) for s in run.scores),
            statistics.fmean(s.recall(5) for s in run.scores),
            statistics.fmean(s.recall(10) for s in run.scores),
            statistics.fmean(s.reciprocal_rank for s in run.scores),
            statistics.fmean(s.ndcg() for s in run.scores),
            _pct(run.latency_ms, 50),
            _pct(run.latency_ms, 95),
        ]
        for name, run in runs.items()
    ]
    headers = ["Stage", "recall@1", "recall@5", "recall@10", "MRR", "NDCG@10", "p50 ms", "p95 ms"]
    floatfmt = [".3f"] * 6 + [".0f", ".0f"]
    print(f"\n## Retrieval stages - {len(questions)} questions, `{store.collection}`\n")
    print(tabulate(rows, headers, tablefmt="github", floatfmt=floatfmt))
    print("\n### recall@5 by question type\n")
    print(by_type_table(questions, runs, "Stage"))

    print("\n### What each step adds (95% bootstrap interval)\n")
    names = list(runs)
    steps = []
    for before, after in itertools.pairwise(names):
        cells: list[str] = [f"{before} -> {after}"]
        for metric in (_recall5, _mrr):
            a = [metric(s) for s in runs[before].scores]
            b = [metric(s) for s in runs[after].scores]
            low, high = bootstrap(a, b)
            mean = statistics.fmean(b) - statistics.fmean(a)
            real = "real" if low > 0 or high < 0 else "noise"
            cells.append(f"{mean:+.3f} [{low:+.3f}, {high:+.3f}] {real}")
        steps.append(cells)
    print(tabulate(steps, ["Step", "△ recall@5", "△ MRR"], tablefmt="github"))

    out = Path(settings.EVAL_DIR) / "results" / f"{date.today()}-pipeline.json"
    save(out, questions, runs, collection=store.collection, golden_set=str(args.golden))
    print(f"\nPer-question results: {out}")
    return 0


if __name__ == "__main__":
    SystemExit(main())
