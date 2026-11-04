"""Day 4: score every chunking strategy against the golden set.

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
from collections.abc import Sequence
from datetime import date
from pathlib import Path

from tabulate import tabulate

from sentinel.config import embedder_from_config, get_settings
from sentinel.eval.golden import DEFAULT_PATH, TARGET, Question, load
from sentinel.eval.retrieval_metrics import KS, Score, score
from sentinel.stores.qdrant_store import QdrantStore

STRATEGIES = ("fixed", "recursive", "semantic")


def evaluate(
    store: QdrantStore,
    questions: Sequence[Question],
    vectors: Sequence[Sequence[float]],
    k: int = 10,
) -> tuple[list[Score], float]:
    """Per-question scores, and mean characters in the top 5 (the context cost)."""
    scores: list[Score] = []
    context: list[int] = []
    for question, vector in zip(questions, vectors, strict=True):
        hits = [
            (str(h.payload["doc_id"]), int(h.payload["start_char"]), int(h.payload["end_char"]))
            for h in store.search(vector, k)
        ]
        scores.append(score(hits, question.answers))
        context.append(sum(end - start for _, start, end in hits[:5]))
    return scores, statistics.fmean(context)


def bootstrap(a: Sequence[float], b: Sequence[float], n: int = 2000) -> tuple[float, float]:
    """95% interval for mean(a) - mean(b), resampling questions as pairs."""
    rng = random.Random(0)
    deltas = [x - y for x, y in zip(a, b, strict=True)]
    means = sorted(statistics.fmean(rng.choices(deltas, k=len(deltas))) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n) - 1]


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
    vectors = [embedder.embed_query(q.question) for q in questions]

    results: dict[str, list[Score]] = {}
    rows: list[list[object]] = []
    for strategy, store in stores.items():
        scores, chars = evaluate(store, questions, vectors)
        results[strategy] = scores
        rows.append(
            [
                strategy,
                *(statistics.fmean(s.recall(k) for s in scores) for k in KS),
                statistics.fmean(s.reciprocal_rank for s in scores),
                statistics.fmean(s.ndcg() for s in scores),
                chars,
            ]
        )
    # "github" is aligned in the console and is also a markdown table for the README.
    headers = ["Chunker", *(f"recall@{k}" for k in KS), "MRR", "NDCG@10", "chars in top 5"]
    floatfmt = [".3f"] * (len(headers) - 1) + [",.0f"]
    print(f"\n## Chunking comparison — {len(questions)} questions, `{embedder.name}`\n")
    print(tabulate(rows, headers, tablefmt="github", floatfmt=floatfmt))

    kinds = [kind for kind in TARGET if any(q.type == kind for q in questions)]
    sizes = {kind: sum(q.type == kind for q in questions) for kind in kinds}
    by_type = [
        [
            strategy,
            *(
                statistics.fmean(
                    s.recall(5) for q, s in zip(questions, scores, strict=True) if q.type == kind
                )
                for kind in kinds
            ),
        ]
        for strategy, scores in results.items()
    ]
    print("\n### recall@5 by question type\n")
    print(
        tabulate(
            by_type,
            ["Chunker", *(f"{kind} (n={sizes[kind]})" for kind in kinds)],
            tablefmt="github",
            floatfmt=".3f",
        )
    )

    by_recall = sorted(results, key=lambda s: statistics.fmean(x.recall(5) for x in results[s]))
    if len(by_recall) > 1:
        best, second = by_recall[-1], by_recall[-2]
        a = [x.recall(5) for x in results[best]]
        b = [x.recall(5) for x in results[second]]
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
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "embedding_model": embedder.name,
                "golden_set": str(args.golden),
                "first_hit_rank": {
                    strategy: {
                        q.id: list(s.first_hit) for q, s in zip(questions, scores, strict=True)
                    }
                    for strategy, scores in results.items()
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nPer-question results: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
