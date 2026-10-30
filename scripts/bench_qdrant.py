from __future__ import annotations

import argparse
import random
import statistics
import time
from collections import Counter
from collections.abc import Sequence
from datetime import date
from typing import Any

from tabulate import tabulate

from sentinel.config import embedder_from_config, get_settings
from sentinel.stores.qdrant_store import Point, QdrantStore

K = 10
HNSW_CONFIGS = ((16, 100), (32, 200))
EF_VALUES = (32, 64, 128)


def run(
    store: QdrantStore, queries: Sequence[Sequence[float]], **params: Any
) -> tuple[list[list[str]], list[float]]:
    results: list[list[str]] = []
    latencies: list[float] = []
    for query in queries:
        started = time.perf_counter()
        hits = store.search(query, K, **params)
        latencies.append((time.perf_counter() - started) * 1000)
        results.append([h.id for h in hits])
    return results, latencies


def recall(got: list[list[str]], truth: list[list[str]]) -> float:
    return statistics.mean(
        len(set(g) & set(t)) / len(t) for g, t in zip(got, truth, strict=True) if t
    )


def pct(latencies: list[float], q: int) -> float:
    return statistics.quantiles(latencies, n=100)[q - 1]


def copy_into(url: str, name: str, points: Sequence[Point], dim: int, **opts: bool) -> QdrantStore:
    store = QdrantStore.connect(url, name)
    store.create(dim, recreate=True, **opts)
    store.upsert_points(points)
    return store


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python scripts/bench_qdrant.py")
    parser.add_argument("--queries", type=int, default=200)
    parser.add_argument(
        "--year",
        type=int,
        default=None,
        help="year for the 8-K filter demo (default: rarest year with 10+ 8-K chunks)",
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    url, name = settings.qdrant_url, settings.qdrant_collection
    main_store = QdrantStore.connect(url, name)
    points = main_store.all_points()
    if len(points) < K:
        print("collection is empty - run scripts/index_corpus.py first")
        return 1
    n, dim = len(points), len(points[0].vector)

    # Queries are real text, not stored vectors: a point's own vector would be
    # its own top hit and flatter every recall number.
    embedder = embedder_from_config(settings)
    sample = random.Random(0).sample(points, min(int(args.queries), n))
    queries = [embedder.embed_query(str(p.payload.get("text", ""))[:200]) for p in sample]
    truth, exact_ms = run(main_store, queries, exact=True)

    print(f"\n## HNSW — {n:,} vectors x {dim} dims, {len(queries)} queries, recall@{K}\n")
    hnsw_headers = [
        "Index",
        "Build",
        "hnsw_ef",
        "p50 ms",
        "p95 ms",
        f"recall@{K}",
    ]

    hnsw_rows: list[list[str]] = [
        [
            "exact (brute force)",
            "—",
            "—",
            f"{pct(exact_ms, 50):.1f}",
            f"{pct(exact_ms, 95):.1f}",
            "1.000",
        ]
    ]
    baseline: tuple[list[list[str]], list[float]] | None = None
    for m, ef_construct in HNSW_CONFIGS:
        bench = copy_into(url, f"{name}_bench", points, dim)
        build_s = bench.build_index(m, ef_construct)
        for ef in EF_VALUES:
            got, ms = run(bench, queries, hnsw_ef=ef)
            if (m, ef) == (16, 64):
                baseline = (got, ms)

            hnsw_rows.append(
                [
                    f"m={m}, ef_construct={ef_construct}",
                    f"{build_s:.1f}s",
                    f"{ef}",
                    f"{pct(ms, 50):.1f}",
                    f"{pct(ms, 95):.1f}",
                    f"{recall(got, truth):.3f}",
                ]
            )
        bench.drop()
    print(
        tabulate(
            hnsw_rows,
            headers=hnsw_headers,
            tablefmt="github",
        )
    )

    print("\n## Scalar int8 quantisation — m=16, ef_construct=100, hnsw_ef=64\n")
    headers = ["Vectors", "Vector memory", "p50 ms", "p95 ms", f"recall@{K}"]
    rows = []
    if baseline:
        got, ms = baseline
        rows.append(
            [
                "float32, in RAM",
                f"{n * dim * 4 / 1e6:.1f} MB",
                f"{pct(ms, 50):.1f}",
                f"{pct(ms, 95):.1f}",
                f"{recall(got, truth):.3f}",
            ]
        )

    quant = copy_into(url, f"{name}_quant", points, dim, on_disk=True, quantize=True)
    quant.build_index()
    for rescore in (False, True):
        got, ms = run(quant, queries, hnsw_ef=64, rescore=rescore)
        label = "int8 + rescore from disk (2x oversampling)" if rescore else "int8, no rescore"

        rows.append(
            [
                label,
                f"{n * dim / 1e6:.1f} MB",
                f"{pct(ms, 50):.1f}",
                f"{pct(ms, 95):.1f}",
                f"{recall(got, truth):.3f}",
            ]
        )

    quant.drop()

    print(tabulate(rows, headers=headers, tablefmt="github"))

    print("\nVector memory is calculated: float32 = n x dim x 4 bytes, int8 = n x dim x 1.")
    print("It excludes the HNSW graph. Latency is client-side over HTTP, round trip included.")
    # --- filtered search -----------------------------------------------------
    years = Counter(
        str(p.payload.get("filing_date", ""))[:4] for p in points if p.payload.get("form") == "8-K"
    )
    eligible = {y: c for y, c in years.items() if c >= K}
    if args.year is None and not eligible:
        print("\n(no year has 10+ 8-K chunks; skipping the filter demo)")
        return 0
    rarest = min(eligible, key=lambda y: eligible[y]) if eligible else ""
    year = int(args.year) if args.year is not None else int(rarest)

    where = main_store.filter(form="8-K", date_from=date(year, 1, 1), date_to=date(year + 1, 1, 1))
    matching = main_store.count(where)
    hits = main_store.search(queries[0], K, where=where)
    exact_hits = main_store.search(queries[0], K, where=where, exact=True)
    all_match = all(
        h.payload.get("form") == "8-K"
        and str(h.payload.get("filing_date", "")).startswith(str(year))
        for h in hits
    )
    overlap = len({h.id for h in hits} & {h.id for h in exact_hits})

    print(f"\n## Filtered search — 8-K filings from {year}\n")
    filter_headers = ["Metric", "Result"]
    filter_rows: list[list[Any]] = [
        [
            "Matching points",
            f"{matching:,} of {n:,} ({matching / n:.1%})",
        ],
        [
            "Returned results",
            f"{len(hits)} of {min(K, matching)} possible",
        ],
        [
            "All results match filter",
            all_match,
        ],
        [
            "Overlap with exact filtered search",
            f"{overlap}/{len(exact_hits)}",
        ],
    ]
    filter_rows.append(
        [
            "Expected results with post-filtering",
            f"~{K * matching / n:.1f}",
        ]
    )

    if not all_match or len(hits) != min(K, matching):
        print("FAIL: filtered search returned non-matching or too few results")
        return 1
    print(tabulate(filter_rows, headers=filter_headers, tablefmt="github"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
