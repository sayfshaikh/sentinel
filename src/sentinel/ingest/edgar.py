from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType

import httpx
from bs4 import BeautifulSoup
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from sentinel.config import get_settings

LOG = logging.getLogger("sentinel.ingest.edgar")

TICKER_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{document}"

DEFAULT_TICKERS: tuple[str, ...] = (
    "AAPL",
    "MSFT",
    "NVDA",
    "GOOGL",
    "AMZN",
    "META",
    "TSLA",
    "AVGO",
    "AMD",
    "INTC",
    "JPM",
    "GS",
    "MS",
    "BAC",
    "C",
    "WFC",
    "XOM",
    "CVX",
    "JNJ",
    "PFE",
    "UNH",
    "WMT",
    "KO",
    "PEP",
    "CAT",
)

DEFAULT_PER_FORM: dict[str, int] = {"10-K": 1, "10-Q": 2, "8-K": 4}

MIN_INTERVAL_S = 0.12  # ~8 req/s


@dataclass
class Filing:
    cik: int
    ticker: str
    form: str
    accession: str
    filing_date: str
    primary_document: str

    @property
    def url(self) -> str:
        return ARCHIVE_URL.format(
            cik=self.cik, accession=self.accession.replace("-", ""), document=self.primary_document
        )


def _html_to_text(raw: bytes) -> str:
    soup = BeautifulSoup(raw, "lxml")
    for tag in soup(["script", "style"]):
        tag.decompose()

    lines = (line.strip() for line in soup.get_text("\n").splitlines())
    return "\n".join(line for line in lines if line)


class EdgarClient:
    "Use as Context Manager: SEC client"

    def __init__(
        self, user_agent: str, raw_dir: Path, min_interval_s: float = MIN_INTERVAL_S
    ) -> None:
        self._client = httpx.Client(
            headers={
                "User-Agent": user_agent,
                "Accept-Encoding": "gzip, deflate",
            },
            timeout=httpx.Timeout(60.0, connect=15.0),
            follow_redirects=True,
        )
        self._raw_dir = raw_dir
        self._min_interval_s = min_interval_s
        self._last_request = 0.0

    def __enter__(self) -> EdgarClient:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._client.close()

    def _throttle(self) -> None:
        remaining = self._min_interval_s - (time.monotonic() - self._last_request)
        if remaining > 0:
            time.sleep(remaining)
        self._last_request = time.monotonic()

    @retry(
        retry=retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError)),
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=1, max=20),
        reraise=True,
    )
    def _get(self, url: str) -> httpx.Response:
        self._throttle()
        response = self._client.get(url)
        response.raise_for_status()
        return response

    def cik_by_ticker(self, tickers: Sequence[str]) -> dict[str, int]:
        payload: dict[str, dict[str, object]] = self._get(TICKER_URL).json()
        lookup = {str(row["ticker"]).upper(): int(str(row["cik_str"])) for row in payload.values()}
        resolved: dict[str, int] = {}
        for ticker in tickers:
            key = ticker.upper()
            if key in lookup:
                resolved[key] = lookup[key]
            else:
                LOG.warning("ticker not found in SEC Index: %s", key)
        return resolved

    def recent_filings(self, ticker: str, cik: int, per_form: dict[str, int]) -> list[Filing]:
        data = self._get(SUBMISSIONS_URL.format(cik=cik)).json()
        recent = data["filings"]["recent"]
        wanted = dict.fromkeys(per_form, 0)
        out: list[Filing] = []

        for i, form in enumerate(recent["form"]):
            if form not in wanted or wanted[form] >= per_form[form]:
                continue
            document = recent["primaryDocument"][i]
            if not document:
                continue
            wanted[form] += 1
            out.append(
                Filing(
                    cik=cik,
                    ticker=ticker,
                    form=form,
                    accession=recent["accessionNumber"][i],
                    filing_date=recent["filingDate"][i],
                    primary_document=document,
                )
            )
            if all(wanted[f] >= per_form[f] for f in wanted):
                break
        return out

    def fetch(self, filing: Filing) -> Path | None:
        target_dir = self._raw_dir / f"{filing.cik:010d}"
        target_dir.mkdir(parents=True, exist_ok=True)
        text_path = target_dir / f"{filing.accession}.txt"
        meta_path = target_dir / f"{filing.accession}.json"

        if text_path.exists() and meta_path.exists():
            LOG.debug("cached %s %s", filing.ticker, filing.accession)
            return text_path

        try:
            response = self._get(filing.url)
        except httpx.HTTPError as exc:
            LOG.warning("failed %s %s: %s", filing.ticker, filing.accession, exc)
            return None

        text = _html_to_text(response.content)
        text_path.write_text(text, encoding="utf-8")
        meta_path.write_text(
            json.dumps(
                {
                    "cik": filing.cik,
                    "ticker": filing.ticker,
                    "accession": filing.accession,
                    "filing_date": filing.filing_date,
                    "source_url": filing.url,
                    "chars": len(text),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        LOG.info(
            "%-6s %-5s %s  %7d chars", filing.ticker, filing.form, filing.filing_date, len(text)
        )
        return text_path


def ingest(
    tickers: Sequence[str],
    per_form: dict[str, int],
    raw_dir: Path,
    user_agent: str,
) -> int:
    fetched = 0
    with EdgarClient(user_agent, raw_dir) as client:
        resolved = client.cik_by_ticker(tickers)
        LOG.info("resolved %d/%d tickers", len(resolved), len(tickers))
        for ticker, cik in resolved.items():
            try:
                filings = client.recent_filings(ticker, cik, per_form)
            except httpx.HTTPError as exc:
                LOG.warning("submissions failed for %s: %s", ticker, exc)
                continue
            for filing in filings:
                if client.fetch(filing) is not None:
                    fetched += 1
    return fetched


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m sentinel.ingest.edgar")
    parser.add_argument(
        "--tickers",
        type=int,
        default=len(DEFAULT_TICKERS),
        help="how many of the built-in ticker list to ingest",
    )
    parser.add_argument(
        "--ticker",
        action="append",
        default=None,
        metavar="SYM",
        help="ingest a specific ticker (repeatable); overrides --tickers",
    )
    parser.add_argument(
        "--per-form",
        type=int,
        default=None,
        help="override the per-form document count for every form",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)-7s %(message)s",
        stream=sys.stderr,
    )

    settings = get_settings()
    explicit: list[str] | None = args.ticker
    selected: Sequence[str] = explicit or DEFAULT_TICKERS[: int(args.tickers)]

    per_form = dict(DEFAULT_PER_FORM)
    if args.per_form is not None:
        per_form = {form: int(args.per_form) for form in per_form}

    count = ingest(selected, per_form, settings.raw_dir, settings.sec_user_agent)
    LOG.info("done: %d documents under %s", count, settings.raw_dir)
    return 0 if count else 1


if __name__ == "__main__":
    raise SystemExit(main())
