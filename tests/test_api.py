from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from sentinel.api.app import app, get_generator, get_retriever
from sentinel.retrieval.base import Passage


class FakeRetriever:
    name = "fake"

    def retrieve(self, query: str, k: int) -> list[Passage]:
        pool = [
            Passage("c1", 1.0, "d/1", 0, 28, "Apple had 166,000 employees."),
            Passage("c2", 0.9, "d/2", 0, 13, "Revenue grew."),
        ]
        return pool[:k]


class FakeGenerator:
    name = "fake-llm"
    context_limit = 10_000

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def generate(self, prompt: str, max_new_tokens: int) -> str:
        return "Apple had 166,000 employees [1]."


@pytest.fixture
def client() -> Iterator[TestClient]:
    app.dependency_overrides[get_retriever] = FakeRetriever
    app.dependency_overrides[get_generator] = FakeGenerator
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_health(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok"}


def test_ask_returns_answer_and_citations(client: TestClient) -> None:
    response = client.post("/ask", json={"question": "How many employees?", "k": 2})
    assert response.status_code == 200
    body = response.json()
    assert "166,000" in body["answer"]
    assert body["generator"] == "fake-llm" and body["retriever"] == "fake"
    assert [c["chunk_id"] for c in body["citations"]] == ["c1"]


def test_ask_rejects_empty_question(client: TestClient) -> None:
    assert client.post("/ask", json={"question": ""}).status_code == 422
