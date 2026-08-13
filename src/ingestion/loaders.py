"""
Document loaders (Step 5, FR-1).

Each loader takes a file path and returns raw text - no chunking here,
that's chunking.py's job. Dispatch is by file extension via
load_document(), which is what pipeline.py (Step 6) actually calls.
"""

from pathlib import Path

from pypdf import PdfReader


def load_txt(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def load_markdown(path: str | Path) -> str:
    # Loaded as raw text on purpose - we chunk the markdown source
    # as-is (headers, bullets, etc. all included) rather than stripping
    # formatting, since that formatting carries semantic structure a
    # retrieval chunk can benefit from.
    return Path(path).read_text(encoding="utf-8")


def load_pdf(path: str | Path) -> str:
    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages)


_LOADERS = {
    ".txt": load_txt,
    ".md": load_markdown,
    ".markdown": load_markdown,
    ".pdf": load_pdf,
}


def load_document(path: str | Path) -> str:
    """
    Dispatches to the right loader based on file extension.
    Raises ValueError for unsupported extensions (FR-1 scopes this to
    PDF, TXT, and Markdown only).
    """
    path = Path(path)
    ext = path.suffix.lower()
    if ext not in _LOADERS:
        raise ValueError(
            f"Unsupported document type '{ext}' for {path}. "
            f"Supported: {sorted(_LOADERS.keys())}"
        )
    return _LOADERS[ext](path)