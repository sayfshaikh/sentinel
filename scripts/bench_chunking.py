from __future__ import annotations

import argparse
import statistics
import time
from collections.abc import Sequence

from sentinel.chunking.base import Chunk, Chunker  # type: ignore[import-untyped]
from sentinel.chunking.fixed import FixedChunker  # type: ignore[import-untyped]
from sentinel.chunking.recursive import RecursiveChunker  # type: ignore[import-untyped]
from sentinel.chunking.semantic import SemanticChunker  # type: ignore[import-untyped] 
from sentinel.config import get_settings  # type: ignore[import-untyped]
from sentinel.embedding.local import SentenceTransformerEmbedder  # type: ignore[import-untyped]
from sentinel.ingest.corpus import Document, iter_documents  # type: ignore[import-untyped]

MODELS = ("BAAI/bge-small-en-v1.5", "BAAI/bge-base-en-v1.5")

def chunk_table(chunkers: Sequence[Chunker], docs: Sequence[Document]) -> list[Chunk]:
    print("\n## Chunking\n")
    print(f"Corpus: {len(docs)} documents, "
          f"{sum(len(d.text) for d in docs):,} characters\n")
    print("| Strategy | Chunks | Mean chars | Median | Min | Max | Wall time |")
    print("|---|---:|---:|---:|---:|---:|---:|")

    widest: list[Chunk] = []
    for chunker in chunkers:
        started = time.perf_counter()
        chunks = [c for doc in docs for c in chunker.chunk(doc)]
        elapsed = time.perf_counter() - started
        lengths = [len(c.text) for c in chunks] or [0]
        print(
            f"| `{chunker.name}` | {len(chunks):,} | {statistics.mean(lengths):.0f} "
            f"| {statistics.median(lengths):.0f} | {min(lengths)} | {max(lengths)} "
            f"| {elapsed:.1f}s |"
        )
        if len(chunks) > len(widest):
            widest = chunks

    return widest


def embed_table(texts: Sequence[str], batch_size: int) -> None:
    print(f"\n## Embedding - {len(texts):,} chunks, batch {batch_size}\n")
    print("| Model | Device | Dim | Warm-up | Encode | Chunks/sec |")
    print("|---|---|---:|---:|---:|---:|")

    for model_name in MODELS:
        embedder = SentenceTransformerEmbedder(model_name, batch_size=batch_size)

        warm_started = time.perf_counter()
        vectors = embedder.embed(list(texts[:8]))
        warm = time.perf_counter() - warm_started

        started = time.perf_counter()
        vectors = embedder.embed(list(texts))
        elapsed = time.perf_counter() - started

        assert len(vectors) == len(texts)
        assert len(vectors[0]) == embedder.dim

        print(
            f"| `{model_name}` | {embedder.device} | {embedder.dim} | {warm:.1f}s "
            f"| {elapsed:.1f}s | {len(texts) / elapsed:.0f} |"
        )

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python scripts/bench_chunking.py")
    parser.add_argument("--docs", type=int, default=10, help="documents to chunk")
    parser.add_argument("--embed", type=int, default=1000, help="chunks to embed")
    args = parser.parse_args(argv)

    settings = get_settings()
    docs = list(iter_documents(settings.raw_dir, limit=int(args.docs)))
    if not docs:
        print(f"no documents under {settings.raw_dir} - run the ingest first")
        return 1

    embedder = SentenceTransformerEmbedder(MODELS[0], batch_size=settings.embedding_batch_size)
    chunkers: list[Chunker] = [
        FixedChunker(settings.chunk_size, settings.chunk_overlap),
        RecursiveChunker(settings.chunk_size, settings.chunk_overlap), 
        SemanticChunker(embedder, percentile=settings.semantic_percentile),
    ]

    chunks = chunk_table(chunkers, docs)
    embed_table([c.text for c in chunks[: int(args.embed)]], settings.embedding_batch_size)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())