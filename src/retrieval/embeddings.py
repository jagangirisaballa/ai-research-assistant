"""Embeddings factory. Pattern: OpenAI Embeddings + Factory."""
from langchain_openai import OpenAIEmbeddings

from src.config import get_settings


def build_embeddings() -> OpenAIEmbeddings:
    settings = get_settings()
    return OpenAIEmbeddings(
        model=settings.openai_embedding_model,
        api_key=settings.openai_api_key,
    )
