"""Graph builder. Pattern: StateGraph with Conditional Edges (LangGraph)."""
from __future__ import annotations

from collections.abc import Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.vectorstores import VectorStore
from langgraph.graph import END, START, StateGraph

from src.graph.nodes import (
    make_responder_node,
    make_retriever_node,
    make_router_node,
    make_web_search_node,
    route_by_intent,
)
from src.graph.state import ResearchState
from src.retrieval.rag_chain import build_llm


def build_research_graph(
    vectorstore: VectorStore,
    llm: BaseChatModel | None = None,
    web_search_fn: Callable[[str], str] | None = None,
):
    """Compose the 4-node research graph and return a compiled, runnable graph.

    Topology:

        START -> router --(intent=docs)----> retriever ---\\
                       --(intent=web)-----> web_search ----> responder -> END
                       --(intent=chitchat)--------------------^
    """
    llm = llm or build_llm()

    graph = StateGraph(ResearchState)
    graph.add_node("router", make_router_node(llm))
    graph.add_node("retriever", make_retriever_node(vectorstore))
    graph.add_node("web_search", make_web_search_node(web_search_fn))
    graph.add_node("responder", make_responder_node(llm))

    graph.add_edge(START, "router")
    graph.add_conditional_edges(
        "router",
        route_by_intent,
        {"retriever": "retriever", "web_search": "web_search", "responder": "responder"},
    )
    graph.add_edge("retriever", "responder")
    graph.add_edge("web_search", "responder")
    graph.add_edge("responder", END)

    return graph.compile()
