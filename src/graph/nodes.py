"""Graph nodes. Each node is a pure function: ResearchState -> partial ResearchState.

Patterns:
- Router Node (LLM-as-classifier)
- Retriever Node (Vector Store)
- Web Search Node (External Tool)
- Responder Node (LLM Synthesis)
"""
from __future__ import annotations

import json
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.vectorstores import VectorStore

from src.graph.state import Intent, ResearchState

_ROUTER_SYSTEM = (
    "You are a routing classifier. Decide how to handle the user's query.\n"
    "Return a JSON object with two keys:\n"
    '  "intent": one of "docs" | "web" | "chitchat"\n'
    '  "reason": short justification (max 15 words)\n'
    "Rules:\n"
    "- 'docs'    -> question likely answerable from the user's indexed documents\n"
    "- 'web'     -> current events, recent news, or general public-web facts\n"
    "- 'chitchat'-> greetings, small talk, meta questions about you\n"
    "Respond with ONLY the JSON object, no prose."
)


def _parse_router_output(raw: str) -> tuple[Intent, str]:
    """Pattern: Defensive Parser — tolerate fenced JSON, stray text, missing keys."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        data = json.loads(text)
        intent = data.get("intent", "chitchat")
        reason = data.get("reason", "")
    except json.JSONDecodeError:
        # Fall back to keyword sniffing — never raise out of the router.
        lowered = text.lower()
        if "web" in lowered:
            intent, reason = "web", "fallback keyword match"
        elif "doc" in lowered:
            intent, reason = "docs", "fallback keyword match"
        else:
            intent, reason = "chitchat", "router parse failed"
    if intent not in ("docs", "web", "chitchat"):
        intent = "chitchat"
    return intent, reason  # type: ignore[return-value]


def make_router_node(llm: BaseChatModel):
    """Pattern: Router Node — closure capturing the LLM dependency."""

    def router_node(state: ResearchState) -> dict[str, Any]:
        query = state.get("query", "")
        if not query:
            return {"intent": "chitchat", "router_reason": "empty query"}
        result = llm.invoke([SystemMessage(content=_ROUTER_SYSTEM), HumanMessage(content=query)])
        intent, reason = _parse_router_output(result.content if hasattr(result, "content") else str(result))
        return {"intent": intent, "router_reason": reason}

    return router_node


def make_retriever_node(vectorstore: VectorStore, k: int = 4):
    """Pattern: Retriever Node."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    def retriever_node(state: ResearchState) -> dict[str, Any]:
        docs = retriever.invoke(state.get("query", ""))
        return {"retrieved_docs": docs}

    return retriever_node


def make_web_search_node(search_fn=None):
    """Pattern: Web Search Node. `search_fn` is injectable for testing."""

    def _default_search(query: str) -> str:
        from langchain_tavily import TavilySearch

        from src.config import get_settings

        tavily = TavilySearch(max_results=5, tavily_api_key=get_settings().tavily_api_key)
        return str(tavily.invoke({"query": query}))

    fn = search_fn or _default_search

    def web_search_node(state: ResearchState) -> dict[str, Any]:
        results = fn(state.get("query", ""))
        return {"web_results": results}

    return web_search_node


_RESPONDER_SYSTEM = (
    "You are a careful research assistant. Synthesise a concise, well-grounded answer "
    "using ONLY the context provided. If the context is empty, say so honestly."
)


def make_responder_node(llm: BaseChatModel):
    """Pattern: Responder Node — synthesises final answer from whichever context is populated."""

    def responder_node(state: ResearchState) -> dict[str, Any]:
        query = state.get("query", "")
        intent = state.get("intent", "chitchat")

        if intent == "docs":
            docs = state.get("retrieved_docs", [])
            context = "\n\n---\n\n".join(d.page_content for d in docs) if docs else ""
            context_label = "Indexed documents"
        elif intent == "web":
            context = state.get("web_results", "") or ""
            context_label = "Web search results"
        else:
            context = ""
            context_label = "No retrieval (chitchat)"

        user_block = f"Query: {query}\n\n[{context_label}]\n{context}" if context else f"Query: {query}\n\n(No external context.)"
        result = llm.invoke([SystemMessage(content=_RESPONDER_SYSTEM), HumanMessage(content=user_block)])
        answer = result.content if hasattr(result, "content") else str(result)
        return {
            "answer": answer,
            "messages": [HumanMessage(content=query), AIMessage(content=answer)],
        }

    return responder_node


def route_by_intent(state: ResearchState) -> str:
    """Pattern: Conditional Edge Function — maps state.intent to next node name."""
    intent = state.get("intent", "chitchat")
    if intent == "docs":
        return "retriever"
    if intent == "web":
        return "web_search"
    return "responder"
