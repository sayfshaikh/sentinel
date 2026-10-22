from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from pydantic import BaseModel, ConfigDict, ValidationError

LOG = logging.getLogger("sentinel.ingest.corpus")


class FilingMeta(BaseModel):

    model_config = ConfigDict(extra="ignore", frozen=True)

    cik: int
    ticker: str
    form: str
    accession: str
    filing_date: date
    source_url: str

    def as_payload(self) -> dict[str, str]:
        return {
            key: str(value)
            for key, value in self.model_dump().items()
        }


@dataclass(frozen=True, slots=True)
class Document:
    doc_id: str
    text: str
    meta: dict[str, str] = field(default_factory=dict)


def iter_documents(raw_dir: Path, limit: int | None = None) -> Iterator[Document]:
    paths = sorted(raw_dir.glob("*/*.txt"))
    if limit is not None:
        paths = paths[:limit]

    for text_path in paths:
        meta_path = text_path.with_suffix(".json")
        if not meta_path.exists():
            LOG.warning("skipping %s: no sidecar .json (interrupted download?)", text_path)
            continue

        try:
            meta = FilingMeta.model_validate_json(meta_path.read_bytes())
        except ValidationError as exc:
            raise ValueError(f"invalid sidecar {meta_path}: {exc}") from exc

        yield Document(
            doc_id=f"{text_path.parent.name}/{text_path.stem}",
            text=text_path.read_text(encoding="utf-8"),
            meta=meta.as_payload(),
        )