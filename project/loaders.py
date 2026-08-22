"""Document loaders — turn files on disk into ``Document`` objects with metadata.

This is the **Metadata** part of ingestion. Loading a file is also the moment
where we attach structured metadata to the text: where it came from, what it is
called, and so on. That metadata travels with every chunk (see
``IngestionCore``) and later lets retrieval filter results — e.g. "only search
HR policies" or "only documents from 2023".

Supports plain text (``.txt`` / ``.md``) and ``.pdf``. Text formats are read
directly; PDFs have their text layer extracted with ``pypdf``. Everything funnels
through ``document_from_text`` so the metadata handling stays in one place.
"""

from __future__ import annotations

import io
from pathlib import Path

from models import Document

SUPPORTED_SUFFIXES = {".txt", ".md", ".pdf"}


def load_document(path: str | Path) -> Document:
    """Read a ``.txt``/``.md``/``.pdf`` file into a ``Document`` with derived metadata.

    Metadata attached (the required fields from ``metadata_schema``):
        - ``source``: the file name, e.g. ``"hr_policy.md"`` — provenance, and a
          handy thing to filter on.
        - ``document_title``: the first Markdown ``# heading`` if present,
          otherwise the file stem (``hr_policy`` -> ``"Hr Policy"``).

    Args:
        path: path to the file.

    Returns:
        A ``Document`` whose ``text`` is the file contents and whose ``metadata``
        carries ``source`` + ``document_title``.
    """
    path = Path(path)
    return document_from_bytes(path.name, path.read_bytes())


def document_from_bytes(filename: str, data: bytes) -> Document:
    """Build a ``Document`` from raw *bytes* + a file name (no disk access).

    This is the path used for **uploads**: a web request hands us the file bytes
    and the original name. It picks the right reader from the extension — plain
    UTF-8 decode for ``.txt``/``.md``, ``pypdf`` text extraction for ``.pdf`` —
    then funnels into ``document_from_text`` for the shared metadata handling.

    Raises ``ValueError`` (which the API maps to HTTP 400) for an unsupported
    type, non-UTF-8 text, or an unreadable PDF.
    """
    name = Path(filename)
    suffix = name.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(
            f"Unsupported file type '{name.suffix}'. "
            f"Supported: {sorted(SUPPORTED_SUFFIXES)}"
        )

    if suffix == ".pdf":
        text = _extract_pdf(data, name.name)
    else:
        # UnicodeDecodeError is a subclass of ValueError, so callers catching
        # ValueError get a clean error for non-text uploads too.
        text = data.decode("utf-8")

    return document_from_text(filename, text)


def document_from_text(filename: str, text: str) -> Document:
    """Build a ``Document`` from already-extracted text + a file *name*.

    The shared tail of every loader: attaches ``source`` (the file name) and
    ``document_title`` (first Markdown ``# heading``, else the title-cased stem),
    so a file produces the same metadata however it was read.

    Args:
        filename: the original file name, e.g. ``"hr_policy.md"``. Used for the
            ``source`` metadata and to derive a fallback title from the stem.
        text: the full document contents.
    """
    name = Path(filename)
    if name.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise ValueError(
            f"Unsupported file type '{name.suffix}'. "
            f"Supported: {sorted(SUPPORTED_SUFFIXES)}"
        )

    metadata = {
        "source": name.name,
        "document_title": _derive_title(text, name.stem),
    }
    return Document(text=text, metadata=metadata)


def _extract_pdf(data: bytes, filename: str) -> str:
    """Extract the text layer of a PDF with ``pypdf``.

    Note this reads the *text layer*: scanned/image-only PDFs have none, so the
    result may be empty (the document then yields no chunks). OCR is out of scope.
    """
    from pypdf import PdfReader  # imported lazily so text-only use needs no pypdf

    try:
        reader = PdfReader(io.BytesIO(data))
        pages = [page.extract_text() or "" for page in reader.pages]
    except Exception as exc:  # noqa: BLE001 - surface any pypdf failure as ValueError
        raise ValueError(f"Could not read PDF '{filename}': {exc}") from exc
    return "\n\n".join(pages).strip()


def load_directory(path: str | Path) -> list[Document]:
    """Load every supported file in a directory (sorted by name for determinism)."""
    path = Path(path)
    if not path.is_dir():
        raise ValueError(f"Not a directory: {path}")

    documents = [
        load_document(file)
        for file in sorted(path.iterdir())
        if file.is_file() and file.suffix.lower() in SUPPORTED_SUFFIXES
    ]
    if not documents:
        raise ValueError(f"No {sorted(SUPPORTED_SUFFIXES)} files found in {path}")
    return documents


def _derive_title(text: str, stem: str) -> str:
    """First Markdown ``# heading`` if present, else a title-cased file stem."""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()
    return stem.replace("_", " ").replace("-", " ").title()
