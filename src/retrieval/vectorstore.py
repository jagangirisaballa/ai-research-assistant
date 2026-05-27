"""FAISS vector store with on-disk persistence. Pattern: Repository + Idempotent Build."""
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.config import get_settings
from src.ingestion.chunker import chunk_documents
from src.ingestion.loader import load_directory
from src.retrieval.embeddings import build_embeddings


def _index_files_exist(path: Path) -> bool:
    return (path / "index.faiss").exists() and (path / "index.pkl").exists()


def build_vectorstore(
    docs: list[Document] | None = None,
    embeddings: Embeddings | None = None,
    persist: bool = True,
) -> FAISS:
    """Build a FAISS store from documents (chunks them first). Persists to disk if requested."""
    if not docs:
        raise ValueError("Cannot build vectorstore from empty document list")
    embeddings = embeddings or build_embeddings()
    chunks = chunk_documents(docs)
    store = FAISS.from_documents(chunks, embeddings)
    if persist:
        path = get_settings().vectorstore_dir
        path.mkdir(parents=True, exist_ok=True)
        store.save_local(str(path))
    return store


def load_vectorstore(embeddings: Embeddings | None = None) -> FAISS | None:
    """Load a previously-persisted store. Returns None if no index on disk."""
    path = get_settings().vectorstore_dir
    if not _index_files_exist(path):
        return None
    embeddings = embeddings or build_embeddings()
    # allow_dangerous_deserialization: required by FAISS loader; safe for indices we wrote ourselves.
    return FAISS.load_local(str(path), embeddings, allow_dangerous_deserialization=True)


def get_or_build_vectorstore() -> FAISS:
    """Idempotent entrypoint: load if present, otherwise build from sample_docs/."""
    embeddings = build_embeddings()
    existing = load_vectorstore(embeddings)
    if existing is not None:
        return existing
    docs = load_directory(get_settings().sample_docs_dir)
    if not docs:
        raise FileNotFoundError(
            f"No documents found in {get_settings().sample_docs_dir}. Drop PDFs/txt in there."
        )
    return build_vectorstore(docs, embeddings=embeddings, persist=True)
