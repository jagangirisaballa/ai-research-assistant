"""Document loading. Pattern: Document Loader (LangChain)."""
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document


class UnsupportedFileTypeError(ValueError):
    """Raised when a file extension has no registered loader."""


# Pattern: Loader Registry — maps extension -> loader factory
_LOADERS = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".md": TextLoader,
}


def load_file(path: Path) -> list[Document]:
    """Load a single file into LangChain Documents."""
    suffix = path.suffix.lower()
    loader_cls = _LOADERS.get(suffix)
    if loader_cls is None:
        raise UnsupportedFileTypeError(f"No loader for {suffix} ({path})")
    return loader_cls(str(path)).load()


def load_directory(directory: Path) -> list[Document]:
    """Load every supported file in a directory. Idempotent: skips unsupported files silently."""
    if not directory.exists():
        return []
    docs: list[Document] = []
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in _LOADERS:
            docs.extend(load_file(path))
    return docs
