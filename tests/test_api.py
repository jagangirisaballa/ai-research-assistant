"""FastAPI endpoint tests — graph is mocked, no LLM / network calls."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document

from src.api import dependencies
from src.api.main import create_app


class _FakeGraph:
    """Mimics the .invoke(state) contract of a compiled LangGraph."""

    def __init__(self, response: dict):
        self._response = response

    def invoke(self, state):  # noqa: ARG002
        return self._response


@pytest.fixture(autouse=True)
def _reset():
    dependencies.reset_state()
    yield
    dependencies.reset_state()


@pytest.fixture
def client():
    return TestClient(create_app())


def test_health_no_vectorstore(client, monkeypatch):
    monkeypatch.setattr(dependencies, "load_vectorstore", lambda: None)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "vectorstore_ready": False}


def test_query_returns_answer_with_sources(client):
    dependencies.set_graph_for_testing(
        _FakeGraph(
            {
                "answer": "Paris.",
                "intent": "docs",
                "router_reason": "matches indexed geography content",
                "retrieved_docs": [Document(page_content="Capital of France is Paris.", metadata={"src": "geo"})],
            }
        )
    )
    r = client.post("/query", json={"query": "What is the capital of France?"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "Paris."
    assert body["intent"] == "docs"
    assert len(body["sources"]) == 1
    assert body["sources"][0]["metadata"]["src"] == "geo"


def test_query_validation_rejects_empty(client):
    r = client.post("/query", json={"query": ""})
    assert r.status_code == 422


def test_query_graph_build_failure_returns_503(client, monkeypatch):
    def _boom():
        raise RuntimeError("simulated openai outage")

    monkeypatch.setattr(dependencies, "get_graph", _boom)
    r = client.post("/query", json={"query": "hello"})
    assert r.status_code == 503
    assert "simulated openai outage" in r.json()["detail"]


def test_query_missing_index_returns_404(client, monkeypatch):
    def _missing():
        raise FileNotFoundError("No documents in data/sample_docs")

    monkeypatch.setattr(dependencies, "get_graph", _missing)
    r = client.post("/query", json={"query": "hello"})
    assert r.status_code == 404


def test_ingest_empty_dir_returns_404(client, monkeypatch, tmp_path):
    from src.config import get_settings

    settings = get_settings()
    monkeypatch.setattr(type(settings), "sample_docs_dir", property(lambda _self: tmp_path))
    r = client.post("/ingest")
    assert r.status_code == 404
