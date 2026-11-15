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
    from sentinel.generate.base import Generator
    from sentinel.retrieval.base import Retriever

BASE_DIR = Path(__file__).resolve().parents[2]

Stage = Literal["dense", "hybrid", "rerank"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="SENTINEL_", extra="ignore")

    # Base Directories
    ROOT_DIR: Path = BASE_DIR
    DATA_DIR: Path = BASE_DIR / "data"
    EVAL_DIR: Path = BASE_DIR / "src" / "sentinel" / "eval"

    sec_user_agent: str = Field(
        ..., description="SEC fair-access requires 'Name email@example.com'."
    )
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "filings"
    postgres_dsn: str = "postgresql://sentinel:sentinel@localhost:5432/sentinel"

    # Chunking
    chunker: Literal["fixed", "recursive", "semantic"] = "recursive"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    semantic_percentile: float = 90.0

    # Embedding
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_batch_size: int = 64

    # Hybrid retrieval and reranking
    bm25_avg_len: float = 160.0
    hybrid_candidates: int = 50
    reranker_model: str = "BAAI/bge-reranker-base"
    reranker_candidates: int = 20

    # Generation
    generator: Literal["local", "api"] = "local"
    generator_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    generator_context: int = 4096
    generator_max_new_tokens: int = 512
    generator_temperature: float = 0.0
    generator_api_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai"
    generator_api_model: str = "gemini-3.8-flash"
    generator_api_key: str = ""

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
        return self.DATA_DIR / "raw"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]


def embedder_from_config(settings: Settings | None = None) -> Embedder:
    from sentinel.embedding.local import SentenceTransformerEmbedder

    s = settings or get_settings()
    return SentenceTransformerEmbedder(
        model_name=s.embedding_model, batch_size=s.embedding_batch_size
    )


def chunker_from_config(
    settings: Settings | None = None, embedder: Embedder | None = None
) -> Chunker:
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


def retriever_from_config(
    stage: Stage = "rerank",
    settings: Settings | None = None,
    embedder: Embedder | None = None,
) -> Retriever:

    from sentinel.embedding.bm25 import BM25Encoder
    from sentinel.retrieval.dense import DenseRetriever
    from sentinel.retrieval.hybrid import HybridRetriever
    from sentinel.retrieval.rerank import CrossEncoderReranker, RerankRetriever
    from sentinel.retrieval.sparse import SparseRetriever
    from sentinel.stores.qdrant_store import QdrantStore

    s = settings or get_settings()
    store = QdrantStore.connect(s.qdrant_url, s.qdrant_collection)
    dense = DenseRetriever(store, embedder or embedder_from_config(s))
    if stage == "dense":
        return dense
    bm25 = SparseRetriever(store, BM25Encoder(avg_len=s.bm25_avg_len))
    hybrid = HybridRetriever([dense, bm25], candidates=s.hybrid_candidates)
    if stage == "hybrid":
        return hybrid
    return RerankRetriever(
        hybrid, CrossEncoderReranker(s.reranker_model), candidates=s.reranker_candidates
    )


def generator_from_config(settings: Settings | None = None) -> Generator:
    s = settings or get_settings()
    if s.generator == "api":
        from sentinel.generate.cloud import ChatAPIGenerator

        if not s.generator_api_key:
            raise ValueError(
                "SENTINEL_GENERATOR=api needs SENTINEL_GENERATOR_API_KEY "
                "(free from Google AI Studio)"
            )
        return ChatAPIGenerator(
            base_url=s.generator_api_base_url,
            api_key=s.generator_api_key,
            model=s.generator_api_model,
            context_limit=s.generator_context,
            temperature=s.generator_temperature,
        )
    from sentinel.generate.local import HFGenerator

    return HFGenerator(
        model_name=s.generator_model,
        context_limit=s.generator_context,
        temperature=s.generator_temperature,
    )
