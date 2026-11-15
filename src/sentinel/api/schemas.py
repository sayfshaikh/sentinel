from __future__ import annotations

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    k: int = Field(default=5, ge=1, le=20)


class CitationModel(BaseModel):
    marker: int
    chunk_id: str
    doc_id: str
    start_char: int
    end_char: int


class AskResponse(BaseModel):
    answer: str
    citations: list[CitationModel]
    retriever: str
    generator: str
