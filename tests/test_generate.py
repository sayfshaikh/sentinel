from __future__ import annotations

from sentinel.generate.prompt import PROMPT_BUDGET, build_prompt, citations_for
from sentinel.retrieval.base import Passage


class WordGen:
    def __init__(self, context_limit: int) -> None:
        self.name = "word"
        self.context_limit = context_limit

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def generate(self, prompt: str, max_new_token: int) -> str:
        return ""


def _passage(cid: str, text: str) -> Passage:
    return Passage(cid, 1.0, f"d/{cid}", 0, len(text), text)


def test_build_prompt_stays_under_budget() -> None:
    passages = [_passage(str(i), "word " * 50) for i in range(20)]
    gen = WordGen(context_limit=200)
    prompt, kept = build_prompt("How many?", passages, gen)
    assert gen.count_tokens(prompt) <= int(PROMPT_BUDGET * gen.context_limit)
    assert 0 < len(kept) < len(passages)
    assert [p.chunk_id for p in kept] == [str(i) for i in range(len(kept))]


def test_citations_map_markers_back_to_chunks() -> None:
    kept = [_passage("a", "x"), _passage("b", "y"), _passage("c", "z")]
    cites = citations_for("The figure is 166,000 [2], up from last year [2].", kept)

    assert [c.marker for c in cites] == [2]
    assert cites[0].chunk_id == "b" and cites[0].doc_id == "d/b"


def test_citations_fall_back_to_all_when_none_cited() -> None:
    kept = [_passage("a", "x"), _passage("b", "y")]
    cites = citations_for("No markers here.", kept)
    assert [c.chunk_id for c in cites] == ["a", "b"]


def test_citations_ignore_out_of_range_markers() -> None:
    kept = [_passage("a", "x")]
    cites = citations_for("see [5]", kept)

    assert [c.chunk_id for c in cites] == ["a"]
