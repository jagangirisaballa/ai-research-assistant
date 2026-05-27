"""LangGraph multi-agent graph tests — fake LLM, deterministic embeddings, no network."""
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from src.graph.builder import build_research_graph
from src.graph.nodes import _parse_router_output
from tests.test_rag import _DeterministicEmbeddings


def _store() -> FAISS:
    docs = [
        Document(page_content="The capital of France is Paris.", metadata={"src": "geo"}),
        Document(page_content="LangChain is a Python framework for LLM apps.", metadata={"src": "lc"}),
    ]
    return FAISS.from_documents(docs, _DeterministicEmbeddings())


# --- Router parser unit tests ---

def test_parser_clean_json():
    intent, reason = _parse_router_output('{"intent": "docs", "reason": "ok"}')
    assert intent == "docs"
    assert reason == "ok"


def test_parser_fenced_json():
    intent, _ = _parse_router_output('```json\n{"intent": "web", "reason": "x"}\n```')
    assert intent == "web"


def test_parser_invalid_falls_back():
    intent, _ = _parse_router_output("totally not json")
    assert intent == "chitchat"


def test_parser_invalid_intent_value_clamped():
    intent, _ = _parse_router_output('{"intent": "bogus", "reason": "x"}')
    assert intent == "chitchat"


# --- End-to-end graph branch tests ---

def _run_branch(router_intent: str, responder_text: str, query: str, web_fn=None):
    store = _store()
    # Router LLM and responder LLM share the same fake model — first call = router JSON,
    # second call = responder synthesis.
    llm = FakeListChatModel(
        responses=[f'{{"intent": "{router_intent}", "reason": "test"}}', responder_text]
    )
    graph = build_research_graph(store, llm=llm, web_search_fn=web_fn)
    return graph.invoke({"query": query})


def test_graph_routes_to_docs():
    final = _run_branch("docs", "Paris is the capital.", "What is the capital of France?")
    assert final["intent"] == "docs"
    assert final["retrieved_docs"]
    assert "Paris" in final["answer"]


def test_graph_routes_to_web():
    final = _run_branch(
        "web",
        "Today's headline is X.",
        "What happened in the news today?",
        web_fn=lambda q: "Mocked web result for: " + q,
    )
    assert final["intent"] == "web"
    assert final["web_results"].startswith("Mocked web result")
    assert "X" in final["answer"]


def test_graph_routes_to_chitchat_skips_retrieval():
    final = _run_branch("chitchat", "Hello there!", "hi")
    assert final["intent"] == "chitchat"
    assert "retrieved_docs" not in final or not final.get("retrieved_docs")
    assert "web_results" not in final or not final.get("web_results")
    assert "Hello" in final["answer"]


def test_graph_emits_messages():
    final = _run_branch("chitchat", "Hi back.", "hi")
    msgs = final.get("messages", [])
    # add_messages reducer should produce HumanMessage + AIMessage
    assert len(msgs) == 2
    assert msgs[0].content == "hi"
    assert msgs[1].content == "Hi back."
