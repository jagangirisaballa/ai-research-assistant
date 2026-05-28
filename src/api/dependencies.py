"""App-level singletons. Pattern: Lazy Singleton + Dependency Injection (FastAPI)."""
from __future__ import annotations

from threading import Lock
from typing import Any

from src.graph.builder import build_research_graph
from src.retrieval.vectorstore import get_or_build_vectorstore, load_vectorstore

_lock = Lock()
_state: dict[str, Any] = {"graph": None, "vectorstore": None}


def reset_state() -> None:
    """Test hook — clears cached singletons."""
    with _lock:
        _state["graph"] = None
        _state["vectorstore"] = None


def get_vectorstore_or_none():
    """Returns the on-disk vectorstore if present, else None. Does NOT build."""
    if _state["vectorstore"] is not None:
        return _state["vectorstore"]
    with _lock:
        if _state["vectorstore"] is None:
            _state["vectorstore"] = load_vectorstore()
        return _state["vectorstore"]


def get_graph():
    """Build (or return cached) compiled graph. Raises if no vectorstore on disk."""
    if _state["graph"] is not None:
        return _state["graph"]
    with _lock:
        if _state["graph"] is None:
            store = _state["vectorstore"] or load_vectorstore()
            if store is None:
                # Build from sample_docs as a last resort — raises FileNotFoundError if empty.
                store = get_or_build_vectorstore()
            _state["vectorstore"] = store
            _state["graph"] = build_research_graph(store)
        return _state["graph"]


def set_graph_for_testing(graph) -> None:
    """Test hook — inject a pre-built graph without touching disk / OpenAI."""
    with _lock:
        _state["graph"] = graph
        _state["vectorstore"] = object()  # marker so health reports ready
