import dataclasses
from functools import lru_cache
from typing import Annotated

from fastapi import Depends, FastAPI

from sentinel.api.schemas import AskRequest, AskResponse, CitationModel
from sentinel.config import generator_from_config, get_settings, retriever_from_config
from sentinel.generate.base import Generator
from sentinel.generate.prompt import build_prompt, citations_for
from sentinel.retrieval.base import Retriever


@lru_cache(maxsize=1)
def get_retriever() -> Retriever:
    return retriever_from_config("rerank")


@lru_cache(maxsize=1)
def get_generator() -> Generator:
    return generator_from_config()


app = FastAPI(title="Sentinel")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ask")
def ask(
    request: AskRequest,
    retriever: Annotated[Retriever, Depends(get_retriever)],
    generator: Annotated[Generator, Depends(get_generator)],
) -> AskResponse:
    passages = retriever.retrieve(request.question, request.k)
    prompt, kept = build_prompt(request.question, passages, generator)
    answer = generator.generate(prompt, get_settings().generator_max_new_tokens)
    citations = citations_for(answer, kept)

    return AskResponse(
        answer=answer,
        citations=[CitationModel(**dataclasses.asdict(c)) for c in citations],
        retriever=retriever.name,
        generator=generator.name,
    )
