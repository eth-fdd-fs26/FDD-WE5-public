from typing import TypedDict, Literal


DocumentType = Literal["regulation", "internal_policy", "guideline"]
Jurisdiction = Literal["EU", "CH", "Global", "US"]


class ChunkMetadata(TypedDict, total=False):
    # --- Required fields ---
    text: str                   # the chunk text
    source: str                 # filename, e.g. "gdpr_full_text.pdf"
    document_title: str         # human-readable, e.g. "General Data Protection Regulation"
    # document_type: DocumentType # broad category for filtering
    # department: str             # owning team, e.g. "HR", "Finance", "Legal", "IT", "All"
    # jurisdiction: str           # geographic/regulatory scope: "EU", "CH", "Global", "US"
    # year: int                   # effective or publication year

    # --- Optional fields (fill in what the PDF extractor can provide) ---
    page_number: int            # PDF page number (if extractable)
    section_title: str          # heading this chunk falls under, e.g. "Article 17 – Right to Erasure"
    article_reference: str      # specific article/clause, e.g. "Article 17", "Section 3.2"
    topic_tags: list[str]       # semantic labels, e.g. ["data_protection", "right_to_erasure"]


REQUIRED_FIELDS: frozenset[str] = frozenset({
    "text",
    "source",
    "document_title",
    # "document_type",
    # "department",
    # "jurisdiction",
    # "year",
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
    """
    Check a payload dict against the schema. Returns a list of error strings.
    Empty list means the payload is valid.
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
