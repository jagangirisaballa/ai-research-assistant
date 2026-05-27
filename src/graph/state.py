"""Graph state. Pattern: TypedDict State Container (LangGraph)."""
from typing import Annotated, Literal, TypedDict

from langchain_core.documents import Document
from langgraph.graph.message import add_messages

Intent = Literal["docs", "web", "chitchat"]


class ResearchState(TypedDict, total=False):
    """Shared state across all graph nodes.

    Fields are intentionally optional (total=False) so each node only writes what it owns —
    LangGraph merges partial dicts into the running state.
    """

    # Inputs
    query: str

    # Router output
    intent: Intent
    router_reason: str

    # Retrieval / search outputs
    retrieved_docs: list[Document]
    web_results: str

    # Final synthesis
    answer: str

    # Optional chat-style transcript. Pattern: Reducer-merged message list.
    messages: Annotated[list, add_messages]
