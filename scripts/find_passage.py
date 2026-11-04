"""Find answer passages for the golden set by keyword.

Deliberately not by the retriever: labelling answers with the system under test
biases the evaluation toward it.

    uv run python scripts/find_passage.py "single source" supplier --ticker AAPL
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from sentinel.config import get_settings
from sentinel.ingest.corpus import iter_documents


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python scripts/find_passage.py")
    parser.add_argument("terms", nargs="+", help="every term must appear in the same paragraph")
    parser.add_argument("--ticker")
    parser.add_argument("--form", help="10-K, 10-Q or 8-K")
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args(argv)

    terms = [t.lower() for t in args.terms]
    shown = 0
    for doc in iter_documents(get_settings().raw_dir):
        if args.ticker and doc.meta.get("ticker") != args.ticker.upper():
            continue
        if args.form and doc.meta.get("form") != args.form.upper():
            continue
        # edgar.py writes one paragraph per line, so a line is a natural passage.
        for line in doc.text.splitlines():
            if all(t in line.lower() for t in terms):
                ticker, form, filed = (doc.meta.get(k) for k in ("ticker", "form", "filing_date"))
                print(f"\n{doc.doc_id}   {ticker} {form} {filed}")
                print(f"  {line[:700]}{' …' if len(line) > 700 else ''}")
                shown += 1
                if shown >= args.limit:
                    return 0
    if not shown:
        print("no matches - try fewer or different terms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
