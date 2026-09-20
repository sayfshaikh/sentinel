from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="SENTINEL_", extra="ignore")

    sec_user_agent: str = Field(
        ..., description="SEC fair-access requires 'Name email@example.com'."
    )
    data_dir: Path = Path("data")
    qdrant_url: str = "http://localhost:6333"
    postgres_dsn: str = "postgresql://sentinel:sentinel@localhost:5432/sentinel"

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
