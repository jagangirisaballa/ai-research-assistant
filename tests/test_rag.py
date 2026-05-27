"""RAG chain smoke tests with a fake LLM + in-memory FAISS store."""
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from src.retrieval import rag_chain


class _DeterministicEmbeddings(Embeddings):
    """Tiny deterministic embedding — hashes characters to a fixed-length vector. No network."""

    DIM = 16

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.DIM
        for i, ch in enumerate(text):
            vec[i % self.DIM] += (ord(ch) % 17) / 17.0
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def _make_store() -> FAISS:
    docs = [
        Document(page_content="The capital of France is Paris.", metadata={"src": "geo"}),
        Document(page_content="LangChain is a framework for LLM apps.", metadata={"src": "lc"}),
    ]
    return FAISS.from_documents(docs, _DeterministicEmbeddings())


def test_retrieval_qa_runs(monkeypatch):
    store = _make_store()
    monkeypatch.setattr(rag_chain, "build_llm", lambda **_: FakeListChatModel(responses=["Paris."]))
    chain = rag_chain.build_retrieval_qa(store, k=2)
    result = chain.invoke({"query": "What is the capital of France?"})
    assert "Paris" in result["result"]
    assert "source_documents" in result


def test_conversational_chain_runs(monkeypatch):
    store = _make_store()
    monkeypatch.setattr(
        rag_chain,
        "build_llm",
        lambda **_: FakeListChatModel(responses=["Paris.", "It is in France."]),
    )
    chain = rag_chain.build_conversational_chain(store, k=2)
    first = chain.invoke({"question": "What is the capital of France?"})
    assert "Paris" in first["answer"]
    second = chain.invoke({"question": "Where is it?"})
    assert second["answer"]
