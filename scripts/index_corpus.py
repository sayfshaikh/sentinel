from __future__ import annotations

import argparse
import dataclasses
import time
from collections.abc import Sequence

from sentinel.config import chunker_from_config, embedder_from_config, get_settings
from sentinel.ingest.corpus import iter_documents
from sentinel.ingest.sections import Sections
from sentinel.stores.qdrant_store import QdrantStore


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python scripts/index_corpus.py")
    parser.add_argument("--docs", type=int, default=None, help="index only the first N docs")
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="drop the collection first; needed after changing the chunker or embedding model",
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    embedder = embedder_from_config(settings)
    chunker = chunker_from_config(settings, embedder)
    store = QdrantStore.connect(settings.qdrant_url, settings.qdrant_collection)
    store.create(embedder.dim, recreate=args.recreate)

    docs = total = 0
    embed_s = 0.0
    for doc in iter_documents(settings.raw_dir, limit=args.docs):
        sections = Sections(doc.text)
        chunks = [
            dataclasses.replace(c, meta={**c.meta, "section": sections.at(c.start_char)})
            for c in chunker.chunk(doc)
        ]
        started = time.perf_counter()
        vectors = embedder.embed([c.text for c in chunks])
        embed_s += time.perf_counter() - started
        store.upsert(chunks, vectors)
        docs += 1
        total += len(chunks)
        ticker, form = doc.meta.get("ticker", "?"), doc.meta.get("form", "?")
        print(f"{ticker:6} {form:5} {len(chunks):6,} chunks")

    if not total:
        print(f"nothing to index under {settings.raw_dir} - run the Day 1 ingest first")
        return 1

    build_s = store.build_index()
    print(f"\n{docs} documents -> {total:,} chunks ({chunker.name})")
    print(f"embedding: {total / embed_s:,.0f} chunks/sec ({embedder.name})")
    print(f"HNSW build: {build_s:.1f}s; collection holds {store.count():,} points")
    return 0



if __name__ == "__main__":
    raise SystemExit(main())