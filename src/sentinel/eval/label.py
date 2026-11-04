"""Add golden-set entries without hand-editing JSON. You still choose every question and quote.

    uv run python -m sentinel.eval.label question s01 single "How many employees did Apple have?"
    uv run python -m sentinel.eval.label answer s01 0000320193/0000320193-25-000079 "As of ..."

Every check runs before anything is written, so a rejected command leaves the file as it was.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from sentinel.config import get_settings
from sentinel.eval.golden import DEFAULT_PATH, TARGET, _Answer, find_quote


def _read(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"version": 1, "questions": []}
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


def _write(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m sentinel.eval.label")
    parser.add_argument("--file", type=Path, default=DEFAULT_PATH)
    commands = parser.add_subparsers(dest="command", required=True)
    q = commands.add_parser("question", help="add a question; its answers come later")
    q.add_argument("id")
    q.add_argument("type", choices=list(TARGET))
    q.add_argument("text")
    a = commands.add_parser("answer", help="add an answer quote to an existing question")
    a.add_argument("id")
    a.add_argument("doc_id")
    a.add_argument("quote")
    args = parser.parse_args(argv)

    data = _read(args.file)
    questions: list[dict[str, Any]] = data["questions"]
    by_id = {entry["id"]: entry for entry in questions}

    if args.command == "question":
        if args.id in by_id:
            print(f"{args.id} already exists")
            return 1
        questions.append({"id": args.id, "type": args.type, "question": args.text, "answers": []})
        print(f"{args.id}: question added")
    else:
        entry = by_id.get(args.id)
        if entry is None:
            print(f"no question {args.id} - add it first with the 'question' command")
            return 1
        try:
            answer = _Answer(doc_id=args.doc_id, quote=args.quote)  # the loader's own rules
        except ValidationError as exc:
            print(exc)
            return 1
        doc = get_settings().raw_dir / f"{answer.doc_id}.txt"
        if not doc.exists():
            print(f"no such document {doc}")
            return 1
        found = find_quote(doc.read_text(encoding="utf-8"), answer.quote)
        if not found:
            print(f"quote not found in {answer.doc_id} - copy it exactly from find_passage")
            return 1
        entry["answers"].append(answer.model_dump())
        print(f"{args.id}: answer added (found {len(found)}x in {answer.doc_id})")

    _write(args.file, data)
    counts = Counter(entry["type"] for entry in questions)
    print("  ".join(f"{kind} {counts[kind]}/{target}" for kind, target in TARGET.items()))
    if waiting := [entry["id"] for entry in questions if not entry["answers"]]:
        print("still need an answer:", ", ".join(waiting))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
