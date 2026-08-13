"""Tests for the ingestion pipeline, grouped by the function each one validates.

Covers chunking.py, loaders.py, and ingestion_core.py — all of which run fully
offline via MockEmbedder and an in-memory Qdrant (DBManager()).

Run it:

    uv run python -m tests.test_ingestion_core
    # or
    uv run python tests/test_ingestion_core.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

# Make the repo root importable whether run as a module or as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chunking
import loaders
from db_manager import DBManager
from ingestion_core import IngestionCore
from models import Document
from retrieval_core import RetrievalCore
from tests.harness import TestSuite
from tests.mocks import MockEmbedder

suite = TestSuite("Ingestion Core — Test Results")


def _ingestion(strategy: str = "sliding_window", chunk_size: int = 10, overlap: int = 2):
    """A fresh IngestionCore backed by an isolated in-memory Qdrant."""
    db = DBManager()  # in-memory, ephemeral
    core = IngestionCore(
        MockEmbedder([1.0, 0.0, 0.0]),
        db,
        collection_name="test_docs",
        chunk_size=chunk_size,
        overlap=overlap,
        strategy=strategy,
    )
    return core, db


def _make_pdf(text: str) -> bytes:
    """A minimal single-page PDF with one line of extractable text.

    Hand-built (with correct xref offsets) so the PDF tests need no extra
    authoring dependency — just enough for ``pypdf`` to pull the text back out.
    """
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        None,  # filled in below (content stream)
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    stream = b"BT /F1 24 Tf 72 720 Td (" + text.encode("latin-1") + b") Tj ET"
    objects[3] = b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream"

    pdf = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, obj in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf += str(i).encode() + b" 0 obj\n" + obj + b"\nendobj\n"
    xref_pos = len(pdf)
    pdf += b"xref\n0 " + str(len(objects) + 1).encode() + b"\n0000000000 65535 f \n"
    for off in offsets:
        pdf += ("%010d 00000 n \n" % off).encode()
    pdf += b"trailer\n<< /Size " + str(len(objects) + 1).encode() + b" /Root 1 0 R >>\n"
    pdf += b"startxref\n" + str(xref_pos).encode() + b"\n%%EOF"
    return bytes(pdf)


# --------------------------------------------------------------------------- #
# sliding_window
# --------------------------------------------------------------------------- #
@suite.case("sliding_window", "splits words into overlapping windows")
def _():
    text = " ".join(f"w{i}" for i in range(9))  # w0 .. w8
    chunks = chunking.sliding_window(text, chunk_size=4, overlap=1)
    # step = 3 -> windows start at 0, 3, 6
    assert chunks == ["w0 w1 w2 w3", "w3 w4 w5 w6", "w6 w7 w8"], chunks


@suite.case("sliding_window", "consecutive chunks share `overlap` words")
def _():
    text = " ".join(f"w{i}" for i in range(10))
    chunks = chunking.sliding_window(text, chunk_size=5, overlap=2)
    first_tail = chunks[0].split()[-2:]
    second_head = chunks[1].split()[:2]
    assert first_tail == second_head, f"{first_tail} != {second_head}"


@suite.case("sliding_window", "short text yields a single chunk")
def _():
    chunks = chunking.sliding_window("only three words", chunk_size=10, overlap=2)
    assert chunks == ["only three words"], chunks


@suite.case("sliding_window", "empty text yields no chunks")
def _():
    assert chunking.sliding_window("   ", chunk_size=5, overlap=1) == []


@suite.case("sliding_window", "rejects overlap >= chunk_size")
def _():
    try:
        chunking.sliding_window("a b c d", chunk_size=3, overlap=3)
        assert False, "expected ValueError when overlap >= chunk_size"
    except ValueError:
        pass


# --------------------------------------------------------------------------- #
# chunk_text / whole_document
# --------------------------------------------------------------------------- #
@suite.case("whole_document", "returns the entire text as one chunk")
def _():
    text = " ".join(f"w{i}" for i in range(50))
    assert chunking.whole_document(text) == [text]


@suite.case("chunk_text", "dispatches to the whole_document strategy")
def _():
    text = " ".join(f"w{i}" for i in range(50))
    chunks = chunking.chunk_text(text, strategy="whole_document")
    assert chunks == [text], "whole_document strategy should yield exactly one chunk"


@suite.case("chunk_text", "rejects an unknown strategy")
def _():
    try:
        chunking.chunk_text("hello world", strategy="banana")
        assert False, "expected ValueError for unknown strategy"
    except ValueError:
        pass


# --------------------------------------------------------------------------- #
# ingest_document
# --------------------------------------------------------------------------- #
@suite.case("ingest_document", "every returned chunk has an embedding")
def _():
    core, _ = _ingestion(chunk_size=5, overlap=1)
    text = " ".join(f"word{i}" for i in range(20))
    chunks = core.ingest_document(Document(text=text, metadata={"source": "x.md", "document_title": "X"}))
    assert len(chunks) > 1, "a 20-word doc with chunk_size 5 should make several chunks"
    assert all(c.embedding is not None for c in chunks), "all chunks must be embedded"


@suite.case("ingest_document", "chunk indices are sequential from 0")
def _():
    core, _ = _ingestion(chunk_size=5, overlap=1)
    text = " ".join(f"word{i}" for i in range(20))
    chunks = core.ingest_document(Document(text=text, metadata={"source": "x.md", "document_title": "X"}))
    assert [c.index for c in chunks] == list(range(len(chunks))), [c.index for c in chunks]


@suite.case("ingest_document", "chunks are persisted and retrievable from the store")
def _():
    core, db = _ingestion(chunk_size=5, overlap=1)
    text = " ".join(f"word{i}" for i in range(20))
    created = core.ingest_document(Document(text=text, metadata={"source": "x.md", "document_title": "X"}))
    stored = db.get_all_chunks("test_docs")
    assert len(stored) == len(created), f"stored {len(stored)} != created {len(created)}"


@suite.case("ingest_document", "empty document stores nothing")
def _():
    core, db = _ingestion()
    chunks = core.ingest_document(Document(text="   ", metadata={"source": "x.md", "document_title": "X"}))
    assert chunks == [], "empty text should produce no chunks"


@suite.case("ingest_document", "can be called repeatedly without re-creating the collection")
def _():
    core, db = _ingestion(chunk_size=5, overlap=1)
    doc_a = Document(text=" ".join(f"a{i}" for i in range(12)), metadata={"source": "a.md", "document_title": "A"})
    doc_b = Document(text=" ".join(f"b{i}" for i in range(12)), metadata={"source": "b.md", "document_title": "B"})
    core.ingest_document(doc_a)
    core.ingest_document(doc_b)  # would raise if ensure_collection weren't guarded
    assert len(db.get_all_documents("test_docs")) == 2


# --------------------------------------------------------------------------- #
# metadata (carried onto chunks + usable as a retrieval filter)
# --------------------------------------------------------------------------- #
@suite.case("ingest_document", "document metadata is carried onto every chunk")
def _():
    core, db = _ingestion(chunk_size=5, overlap=1)
    text = " ".join(f"word{i}" for i in range(20))
    core.ingest_document(Document(text=text, metadata={"source": "hr.md", "document_title": "HR"}))
    stored = db.get_all_chunks("test_docs")
    assert all(c.metadata.get("source") == "hr.md" for c in stored), "source missing on a chunk"
    assert all(c.metadata.get("document_title") == "HR" for c in stored)


@suite.case("metadata_filter", "narrows retrieval to matching chunks")
def _():
    core, db = _ingestion(chunk_size=5, overlap=1)
    core.ingest_document(Document(text=" ".join(f"a{i}" for i in range(12)), metadata={"source": "a.md", "document_title": "A"}))
    core.ingest_document(Document(text=" ".join(f"b{i}" for i in range(12)), metadata={"source": "b.md", "document_title": "B"}))

    rc = RetrievalCore(MockEmbedder([1.0, 0.0, 0.0]), db.get_all_chunks("test_docs"))
    results = rc.embedding_search("anything", top_k=10, metadata_filter={"source": "a.md"})
    assert results, "filter should still return the matching chunks"
    assert all(c.metadata.get("source") == "a.md" for c, _ in results), "filter leaked other sources"


# --------------------------------------------------------------------------- #
# loaders.load_document
# --------------------------------------------------------------------------- #
@suite.case("load_document", "derives title from the first Markdown heading")
def _():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "policy.md"
        path.write_text("# Travel Policy\n\nSome body text.", encoding="utf-8")
        doc = loaders.load_document(path)
    assert doc.metadata["document_title"] == "Travel Policy", doc.metadata
    assert doc.metadata["source"] == "policy.md", doc.metadata


@suite.case("load_document", "falls back to the file stem when there is no heading")
def _():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "raw_notes.txt"
        path.write_text("just some text, no heading", encoding="utf-8")
        doc = loaders.load_document(path)
    assert doc.metadata["document_title"] == "Raw Notes", doc.metadata


@suite.case("load_document", "rejects unsupported file types")
def _():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "data.docx"
        path.write_text("x", encoding="utf-8")
        try:
            loaders.load_document(path)
            assert False, "expected ValueError for a .docx file"
        except ValueError:
            pass


@suite.case("load_document", "reads the text layer of a PDF")
def _():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "report.pdf"
        path.write_bytes(_make_pdf("Quarterly compliance report"))
        doc = loaders.load_document(path)
    assert "Quarterly compliance report" in doc.text, doc.text
    assert doc.metadata["source"] == "report.pdf", doc.metadata
    assert doc.metadata["document_title"] == "Report", doc.metadata  # stem fallback


# --------------------------------------------------------------------------- #
# loaders.document_from_text  (already-extracted text)
# --------------------------------------------------------------------------- #
@suite.case("document_from_text", "builds a Document from raw text + filename")
def _():
    doc = loaders.document_from_text("hr_policy.md", "# HR Policy\n\nBody text.")
    assert doc.metadata["source"] == "hr_policy.md", doc.metadata
    assert doc.metadata["document_title"] == "HR Policy", doc.metadata
    assert doc.text == "# HR Policy\n\nBody text."


@suite.case("document_from_text", "falls back to the filename stem when there is no heading")
def _():
    doc = loaders.document_from_text("raw_notes.txt", "just text, no heading")
    assert doc.metadata["document_title"] == "Raw Notes", doc.metadata


# --------------------------------------------------------------------------- #
# loaders.document_from_bytes  (the upload path — bytes, no file on disk)
# --------------------------------------------------------------------------- #
@suite.case("document_from_bytes", "decodes uploaded text bytes")
def _():
    doc = loaders.document_from_bytes("notes.md", b"# Notes\n\nhello")
    assert doc.text == "# Notes\n\nhello"
    assert doc.metadata["document_title"] == "Notes", doc.metadata


@suite.case("document_from_bytes", "extracts text from uploaded PDF bytes")
def _():
    doc = loaders.document_from_bytes("policy.pdf", _make_pdf("Remote work is allowed"))
    assert "Remote work is allowed" in doc.text, doc.text
    assert doc.metadata["source"] == "policy.pdf", doc.metadata


@suite.case("document_from_bytes", "rejects unsupported file types")
def _():
    try:
        loaders.document_from_bytes("data.docx", b"x")
        assert False, "expected ValueError for a .docx name"
    except ValueError:
        pass


@suite.case("document_from_bytes", "raises a clear error on an unreadable PDF")
def _():
    try:
        loaders.document_from_bytes("broken.pdf", b"not really a pdf")
        assert False, "expected ValueError for invalid PDF bytes"
    except ValueError as exc:
        assert "broken.pdf" in str(exc), exc


if __name__ == "__main__":
    raise SystemExit(0 if suite.run() else 1)
