"""Tests for db_manager.py — the Qdrant-backed vector store wrapper.

Uses fake (random) vectors since the embedding model isn't wired in here; the
DB layer doesn't care what the numbers are, only that the dimensions match.

Run it:

    uv run python -m tests.test_db_manager
    # or
    uv run python tests/test_db_manager.py
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

# Make the repo root importable whether run as a module or as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from db_manager import DBManager
from models.chunk import Chunk
from models.document import Document

VECTOR_SIZE = 384
COLLECTION = "compliance_documents"

random.seed(0)  # deterministic fake vectors


# --------------------------------------------------------------------------- #
# Fixtures / fakes
# --------------------------------------------------------------------------- #
def _fake_vector() -> list[float]:
    return [random.uniform(-1, 1) for _ in range(VECTOR_SIZE)]


def _fake_chunk(document_id: str, index: int, text: str, embedded: bool = True, **metadata) -> Chunk:
    # Always supply the schema's required fields so payloads are valid.
    base_metadata = {"source": f"{document_id}.pdf", "document_title": document_id.replace("_", " ").title()}
    base_metadata.update(metadata)
    return Chunk(
        document_id=document_id,
        index=index,
        text=text,
        metadata=base_metadata,
        embedding=_fake_vector() if embedded else None,
    )


def _fake_chunks() -> list[Chunk]:
    """Five chunks spread across four documents (hr_policy has two)."""
    return [
        _fake_chunk("hr_policy_2025", 0, "Annual compliance training is due by December 31st."),
        _fake_chunk("aml_guidelines", 0, "Transactions above CHF 10,000 must be reported."),
        _fake_chunk("hr_policy_2025", 1, "Remote work requires manager approval."),
        _fake_chunk("gdpr_full_text", 0, "Article 17 grants the right to erasure."),
        _fake_chunk("vendor_risk_policy", 0, "Vendors undergo a risk assessment before onboarding."),
    ]


def _fresh_db() -> DBManager:
    db = DBManager()
    db.create_collection(collection_name=COLLECTION, vector_size=VECTOR_SIZE)
    return db


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def test_insert_chunks_and_get_all_chunks():
    db = _fresh_db()
    db.insert_chunks(COLLECTION, _fake_chunks())

    chunks = db.get_all_chunks(COLLECTION)
    assert len(chunks) == 5, f"expected 5 chunks, got {len(chunks)}"
    # Embeddings must come back restored, at the right dimension.
    assert all(c.embedding is not None and len(c.embedding) == VECTOR_SIZE for c in chunks), "embeddings not restored"
    # Structural keys must not leak into metadata.
    assert all({"document_id", "index", "text"}.isdisjoint(c.metadata) for c in chunks), "structural keys leaked into metadata"


def test_get_all_documents_is_unique():
    db = _fresh_db()
    db.insert_chunks(COLLECTION, _fake_chunks())

    doc_ids = db.get_all_documents(COLLECTION)
    assert len(doc_ids) == 4, f"expected 4 unique documents, got {len(doc_ids)}"
    assert len(set(doc_ids)) == len(doc_ids), "document ids should be unique"
    assert set(doc_ids) == {"hr_policy_2025", "aml_guidelines", "gdpr_full_text", "vendor_risk_policy"}


def test_insert_document_stores_chunks_and_links_ids():
    db = _fresh_db()
    document = Document(
        text="Full GDPR text ...",
        metadata={"source": "gdpr_full_text.pdf", "document_title": "GDPR"},
    )
    # insert_document trusts the chunks it's given — IngestionCore.ingest_document
    # is the only caller, and it already sets document_id + metadata correctly
    # when it builds them.
    chunks = [
        Chunk(document_id=document.id, index=0, text="Article 17 grants the right to erasure.",
              metadata=dict(document.metadata), embedding=_fake_vector()),
        Chunk(document_id=document.id, index=1, text="Article 18 covers restriction of processing.",
              metadata=dict(document.metadata), embedding=_fake_vector()),
    ]
    db.insert_document(COLLECTION, document, chunks)

    # The document now tracks its chunk ids.
    assert document.chunk_ids == [c.id for c in chunks], "document.chunk_ids not updated"

    stored = db.get_all_chunks(COLLECTION)
    assert len(stored) == 2, f"expected 2 chunks, got {len(stored)}"
    assert all(c.document_id == document.id for c in stored), "document_id not preserved"
    assert all(c.metadata.get("source") == "gdpr_full_text.pdf" for c in stored), "chunk metadata not preserved"


def test_delete_removes_chunk():
    db = _fresh_db()
    db.insert_chunks(COLLECTION, _fake_chunks())
    target = db.get_all_chunks(COLLECTION)[0]

    db.delete(COLLECTION, [target.id])

    remaining = db.get_all_chunks(COLLECTION)
    assert len(remaining) == 4, f"expected 4 chunks after delete, got {len(remaining)}"
    assert target.id not in {c.id for c in remaining}, "deleted chunk still present"


def test_insert_chunks_requires_embedding():
    db = _fresh_db()
    bad = _fake_chunk("hr_policy_2025", 0, "no embedding here", embedded=False)
    try:
        db.insert_chunks(COLLECTION, [bad])
    except ValueError:
        return
    raise AssertionError("inserting a chunk with no embedding should raise ValueError")


def test_insert_empty_chunks_is_noop():
    db = _fresh_db()
    db.insert_chunks(COLLECTION, [])
    assert db.get_all_chunks(COLLECTION) == [], "empty insert should store nothing"


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
TESTS = [
    test_insert_chunks_and_get_all_chunks,
    test_get_all_documents_is_unique,
    test_insert_document_stores_chunks_and_links_ids,
    test_delete_removes_chunk,
    test_insert_chunks_requires_embedding,
    test_insert_empty_chunks_is_noop,
]


def main() -> int:
    passed = 0
    failures: list[str] = []
    for test in TESTS:
        try:
            test()
            passed += 1
            print(f"  PASS  {test.__name__}")
        except Exception as exc:  # noqa: BLE001 - report everything
            failures.append(f"  FAIL  {test.__name__}  -> {type(exc).__name__}: {exc}")

    print()
    for line in failures:
        print(line)
    print(f"\n{passed}/{len(TESTS)} tests passed")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
