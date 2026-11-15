from __future__ import annotations

import re
from collections.abc import Sequence
from typing import TYPE_CHECKING

from sentinel.generate.base import Citation, Generator

if TYPE_CHECKING:
    from sentinel.retrieval.base import Passage

PROMPT_BUDGET = 0.6

_SYSTEM = (
    "You answer questions about SEC filings using only the numbered context below. "
    "Cite every claim with its source marker, like [1] or [2]. If the context does "
    "not contain the answer, say so. Do not use outside knowledge."
)

_MARKER = re.compile(r"\[(\d+)\]")


def _render(question: str, blocks: Sequence[str]) -> str:
    context = "\n\n".join(blocks)
    return f"{_SYSTEM}\n\n{context}\n\nQuestion: {question}\n\nAnswer:"


def build_prompt(
    question: str, passages: Sequence[Passage], generator: Generator
) -> tuple[str, list[Passage]]:
    budget = int(PROMPT_BUDGET * generator.context_limit)
    kept: list[Passage] = []
    blocks: list[str] = []
    for passage in passages:
        block = f"[{len(kept) + 1}] (Chunk {passage.chunk_id})\n{passage.text}"
        if kept and generator.count_tokens(_render(question, [*blocks, block])) > budget:
            break
        kept.append(passage)
        blocks.append(block)
    prompt = _render(question, blocks)
    tokens = generator.count_tokens(prompt)
    assert tokens <= budget, (
        f"prompt is {tokens} tokens, over the {budget}-token budget "
        f"({PROMPT_BUDGET:.0%} of {generator.context_limit}); lower k or raise "
        "SENTINEL_GENERATOR_CONTEXT"
    )
    return prompt, kept


def citations_for(answer: str, kept: Sequence[Passage]) -> list[Citation]:
    used = sorted({n for m in _MARKER.findall(answer) if 1 <= (n := int(m)) <= len(kept)})
    markers = used if used else list(range(1, len(kept) + 1))
    return [
        Citation(
            n,
            kept[n - 1].chunk_id,
            kept[n - 1].doc_id,
            kept[n - 1].start_char,
            kept[n - 1].end_char,
        )
        for n in markers
    ]
