"""FastAPI REST wrapper.

Patterns:
- Application Factory (create_app)
- Structured Error Handler (HTTPException -> ErrorResponse)
- Idempotent Ingest Endpoint
"""
from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from src.api import dependencies
from src.api.schemas import (
    ErrorResponse,
    HealthResponse,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    SourceChunk,
)
from src.config import get_settings
from src.ingestion.chunker import chunk_documents
from src.ingestion.loader import load_directory
from src.retrieval.vectorstore import build_vectorstore

log = logging.getLogger(__name__)


def create_app() -> FastAPI:
    app = FastAPI(
        title="AI Research Assistant",
        description="RAG-powered multi-agent research assistant (LangChain + LangGraph)",
        version="0.1.0",
    )

    @app.exception_handler(HTTPException)
    async def _http_exc(_, exc: HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(error=exc.__class__.__name__, detail=str(exc.detail)).model_dump(),
        )

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        vs = dependencies.get_vectorstore_or_none()
        return HealthResponse(status="ok", vectorstore_ready=vs is not None)

    @app.post("/ingest", response_model=IngestResponse)
    async def ingest() -> IngestResponse:
        """Rebuild the FAISS index from data/sample_docs/. Idempotent — safe to re-run."""
        settings = get_settings()
        docs = load_directory(settings.sample_docs_dir)
        if not docs:
            raise HTTPException(
                status_code=404,
                detail=f"No documents in {settings.sample_docs_dir}. Drop PDFs/txt files there first.",
            )
        try:
            build_vectorstore(docs, persist=True)
        except Exception as exc:
            log.exception("vectorstore build failed")
            raise HTTPException(status_code=503, detail=f"Embedding/build failed: {exc}") from exc
        dependencies.reset_state()  # force graph rebuild next request against fresh store
        chunks = chunk_documents(docs)
        return IngestResponse(
            indexed_files=len({d.metadata.get("source") for d in docs}),
            chunks=len(chunks),
            vectorstore_path=str(settings.vectorstore_dir),
        )

    @app.post("/query", response_model=QueryResponse)
    async def query(req: QueryRequest) -> QueryResponse:
        try:
            graph = dependencies.get_graph()
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            log.exception("graph build failed")
            raise HTTPException(status_code=503, detail=f"Graph unavailable: {exc}") from exc

        try:
            final = graph.invoke({"query": req.query})
        except Exception as exc:
            log.exception("graph execution failed")
            raise HTTPException(status_code=503, detail=f"Upstream LLM/tool failure: {exc}") from exc

        sources = [
            SourceChunk(content=d.page_content, metadata=dict(d.metadata))
            for d in final.get("retrieved_docs", []) or []
        ]
        return QueryResponse(
            answer=final.get("answer", ""),
            intent=final.get("intent", "chitchat"),
            router_reason=final.get("router_reason", ""),
            sources=sources,
        )

    return app


app = create_app()
