"""The contract every stored chunk payload must satisfy.

Metadata is what makes retrieval *precise*: it rides from the loader onto every
chunk, is flattened into the Qdrant payload, and lets a query narrow to one
document ("only search the leave policy") before any scoring happens. This module
is the single place that says which keys must be there.

The required set is deliberately small. Every extra required key is a key the
loader must be able to produce for *every* document — a rule that is easy to
write and painful to keep. The optional keys below are the ones a richer loader
(PDF extraction, a document management system) can supply when it has them.
"""

from typing import Literal, TypedDict

DocumentType = Literal["regulation", "internal_policy", "guideline"]
Jurisdiction = Literal["EU", "CH", "Global", "US"]


class ChunkMetadata(TypedDict, total=False):
    """The shape of a chunk payload. ``total=False`` — see REQUIRED_FIELDS."""

    # --- Required on every chunk (enforced by validate_payload) ---
    text: str  # the chunk text
    source: str  # filename, e.g. "leave_and_absence_policy.md"
    document_title: str  # human-readable, e.g. "Leave and Absence Policy"

    # --- Optional: supplied when the loader can extract them ---
    document_type: DocumentType  # broad category, for filtering
    department: str  # owning team, e.g. "HR", "Finance", "Legal", "IT"
    jurisdiction: Jurisdiction  # geographic/regulatory scope
    year: int  # effective or publication year
    page_number: int  # PDF page number (if extractable)
    section_title: str  # heading this chunk falls under
    article_reference: str  # specific article/clause, e.g. "Article 17"
    topic_tags: list[str]  # semantic labels, e.g. ["data_protection"]


#: Keys that must be present on every payload. Keep this minimal — see the
#: module docstring. Any key in ChunkMetadata may be used as a metadata filter
#: whether or not it is required.
REQUIRED_FIELDS: frozenset[str] = frozenset({
    "text",
    "source",
    "document_title",
})

VALID_DOCUMENT_TYPES: frozenset[str] = frozenset({
    "regulation",
    "internal_policy",
    "guideline",
})

VALID_JURISDICTIONS: frozenset[str] = frozenset({
    "EU",
    "CH",
    "Global",
    "US",
})


def validate_payload(payload: dict) -> list[str]:
    """Check a payload against the schema.

    Returns a list of human-readable error strings; an empty list means valid.
    Called on every write path (``insert_points`` and ``insert_chunks``), so a
    chunk that lost its metadata fails loudly at ingestion instead of quietly
    degrading retrieval later.
    """
    errors: list[str] = []

    missing = REQUIRED_FIELDS - payload.keys()
    if missing:
        errors.append(f"Missing required fields: {sorted(missing)}")

    doc_type = payload.get("document_type")
    if doc_type and doc_type not in VALID_DOCUMENT_TYPES:
        errors.append(
            f"Invalid document_type '{doc_type}'. Must be one of: {sorted(VALID_DOCUMENT_TYPES)}"
        )

    jurisdiction = payload.get("jurisdiction")
    if jurisdiction and jurisdiction not in VALID_JURISDICTIONS:
        errors.append(
            f"Invalid jurisdiction '{jurisdiction}'. Must be one of: {sorted(VALID_JURISDICTIONS)}"
        )

    if "year" in payload and not isinstance(payload["year"], int):
        errors.append(f"'year' must be an int, got {type(payload['year']).__name__}")

    if "chunk_index" in payload and not isinstance(payload["chunk_index"], int):
        errors.append(f"'chunk_index' must be an int, got {type(payload['chunk_index']).__name__}")

    if "topic_tags" in payload and not isinstance(payload["topic_tags"], list):
        errors.append(f"'topic_tags' must be a list, got {type(payload['topic_tags']).__name__}")

    return errors
