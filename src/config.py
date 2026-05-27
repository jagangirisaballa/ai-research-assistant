"""Centralised configuration. Pattern: Settings Singleton (pydantic-settings)."""
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    openai_api_key: str = ""
    tavily_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    vectorstore_path: str = "data/vectorstore"
    sample_docs_path: str = "data/sample_docs"
    chunk_size: int = 1000
    chunk_overlap: int = 150

    @property
    def vectorstore_dir(self) -> Path:
        return Path(self.vectorstore_path)

    @property
    def sample_docs_dir(self) -> Path:
        return Path(self.sample_docs_path)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
