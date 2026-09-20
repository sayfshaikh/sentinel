from collections.abc import Callable

import psycopg
from qdrant_client import QdrantClient

from sentinel.config import get_settings


def check_qdrant(url: str) -> str:
    client = QdrantClient(url=url, timeout=5)
    try:
        collections = client.get_collections().collections
    finally:
        client.close()

    return f"{url} collections={len(collections)}"


def check_postgres(dsn: str) -> str:
    with psycopg.connect(dsn, connect_timeout=5) as conn:
        row = conn.execute("select version()").fetchone()
    if row is None:
        raise RuntimeError("select version() returned no rows")
    return str(row[0]).split(",")[0]


def main() -> int:
    settings = get_settings()

    probes: tuple[tuple[str, Callable[[str], str], str], ...] = (
        ("qdrant", check_qdrant, settings.qdrant_url),
        ("postgres", check_postgres, settings.postgres_dsn),
    )

    failed = 0
    for name, probe, target in probes:
        try:
            detail = probe(target)
        except Exception as exc:
            print(f"{name:<9} FAIL {type(exc).__name__}: {exc}")
            failed += 1
        else:
            print(f"{name:<9} OK {detail}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
