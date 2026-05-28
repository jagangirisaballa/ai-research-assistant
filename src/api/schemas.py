"""API request/response models. Pattern: Pydantic DTO."""
from __future__ import annotations

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)


class SourceChunk(BaseModel):
    content: str
    metadata: dict = Field(default_factory=dict)


class QueryResponse(BaseModel):
    answer: str
    intent: str
    router_reason: str = ""
    sources: list[SourceChunk] = Field(default_factory=list)


class IngestResponse(BaseModel):
    indexed_files: int
    chunks: int
    vectorstore_path: str


class HealthResponse(BaseModel):
    status: str
    vectorstore_ready: bool


class ErrorResponse(BaseModel):
    error: str
    detail: str = ""
