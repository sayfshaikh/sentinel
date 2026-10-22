from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

if TYPE_CHECKING:
    from sentinel.chunking.base import Chunker
    from sentinel.embedding.base import Embedder

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="SENTINEL_", extra="ignore")

    sec_user_agent: str = Field(
        ..., description="SEC fair-access requires 'Name email@example.com'."
    )
    data_dir: Path = Path("data")
    qdrant_url: str = "http://localhost:6333"
    postgres_dsn: str = "postgresql://sentinel:sentinel@localhost:5432/sentinel"

    # Chunking
    chunker: Literal["fixed", "recursive", "semantic"] = "recursive"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    semantic_percentile: float = 90.0

    # Embedding
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_batch_size: int = 64

    @field_validator("sec_user_agent")
    @classmethod
    def _must_identify(cls, v: str) -> str:
        if "@" not in v or len(v.split()) < 2:
            raise ValueError(
                "SENTINEL_SEC_USER_AGENT must look like 'Your Name you@example.com'. "
                "SEC blocks anonymous clients."
            )
        return v

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]

def embedder_from_config(settings: Settings | None = None) -> Embedder:
    from sentinel.embedding.local import SentenceTransformerEmbedder

    s = settings or get_settings()
    return SentenceTransformerEmbedder(
        model_name=s.embedding_model,
        batch_size=s.embedding_batch_size
    )


def chunker_from_config(settings: Settings | None = None, embedder: Embedder | None = None) -> Chunker:
    from sentinel.chunking.fixed import FixedChunker
    from sentinel.chunking.recursive import RecursiveChunker
    from sentinel.chunking.semantic import SemanticChunker

    s = settings or get_settings()

    builders: dict[str, Callable[[], Chunker]] = {
        "fixed": lambda: FixedChunker(s.chunk_size, s.chunk_overlap),
        "recursive": lambda: RecursiveChunker(s.chunk_size, s.chunk_overlap),
        "semantic": lambda: SemanticChunker(
            embedder or embedder_from_config(s),
            percentile=s.semantic_percentile,
        ),
    }

    try:
        return builders[s.chunker]()
    except KeyError:
        raise ValueError(
            f"unknown chunker {s.chunker!r}; expected one of {sorted(builders)}"
        ) from None
    