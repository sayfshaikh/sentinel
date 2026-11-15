from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class Citation:
    marker: int
    chunk_id: str
    doc_id: str
    start_char: int
    end_char: int


@dataclass(frozen=True, slots=True)
class GeneratedAnswer:
    text: str
    citations: list[Citation]


class Generator(Protocol):
    name: str
    context_limit: int

    def count_tokens(self, text: str) -> int: ...

    def generate(self, prompt: str, max_new_tokens: int) -> str: ...
