from __future__ import annotations

import bisect
import re

# Item <number> at the start of a line. [^\S\n] is any whitespace except a
# newline, which includes the non-breaking spaces SEC HTML is full of.
_HEADING = re.compile(
    r"^[^\S\n]*item[^\S\n]+(\d{1,2}[a-c]?(?:\.\d{2})?)\.?(?=\s)",
    re.IGNORECASE | re.MULTILINE,
)


class Sections:
    def __init__(self, text: str) -> None:
        last: dict[str, int] = {}
        for match in _HEADING.finditer(text):
            last[f"Item {match.group(1).upper()}"] = match.start()
        ordered = sorted((start, label) for label, start in last.items())
        self._starts = [start for start, _ in ordered]
        self._labels = [label for _, label in ordered]

    def at(self, offset: int) -> str:
        i = bisect.bisect_right(self._starts, offset) - 1
        return self._labels[i] if i >= 0 else "preamble"
