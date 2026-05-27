"""Tests for ingestion layer — no network, no LLM."""
from pathlib import Path

import pytest
from langchain_core.documents import Document

from src.ingestion.chunker import chunk_documents
from src.ingestion.loader import UnsupportedFileTypeError, load_directory, load_file


def test_load_directory_empty(tmp_path: Path):
    assert load_directory(tmp_path) == []


def test_load_directory_missing(tmp_path: Path):
    assert load_directory(tmp_path / "does-not-exist") == []


def test_load_txt_file(tmp_path: Path):
    f = tmp_path / "sample.txt"
    f.write_text("Hello world. This is a sample document for testing.")
    docs = load_file(f)
    assert len(docs) == 1
    assert "Hello world" in docs[0].page_content


def test_load_unsupported_extension(tmp_path: Path):
    f = tmp_path / "binary.bin"
    f.write_bytes(b"\x00\x01")
    with pytest.raises(UnsupportedFileTypeError):
        load_file(f)


def test_load_directory_skips_unsupported(tmp_path: Path):
    (tmp_path / "a.txt").write_text("alpha content here")
    (tmp_path / "b.bin").write_bytes(b"\x00")
    docs = load_directory(tmp_path)
    assert len(docs) == 1


def test_chunker_splits_long_document():
    long_text = "Sentence. " * 500
    docs = [Document(page_content=long_text, metadata={"source": "test"})]
    chunks = chunk_documents(docs)
    assert len(chunks) > 1
    assert all(c.metadata.get("source") == "test" for c in chunks)
