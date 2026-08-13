"""Tests for rag_core.py, grouped by the function each one validates.

RAGCore is the orchestrator, so these checks focus on *wiring*: how it picks a
storage backend, how ingestion hydrates the retriever, and how answer_query
dispatches to the right strategy and feeds the LLM. Everything runs offline via
MockEmbedder / MockLLM and an in-memory Qdrant.

Run it:

    uv run python -m tests.test_rag_core
    # or
    uv run python tests/test_rag_core.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

# Make the repo root importable whether run as a module or as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from db_manager import DBManager
from models import Document
from rag_core import RAGCore
from models import SearchType
from tests.harness import TestSuite
from tests.mocks import MockEmbedder, MockLLM

suite = TestSuite("RAG Core — Test Results")


def _rag(**kwargs) -> RAGCore:
    """A RAGCore backed by mocks and a fresh in-memory store."""
    return RAGCore(MockEmbedder([1.0, 0.0, 0.0]), MockLLM(), **kwargs)


def _doc(text: str, **metadata) -> Document:
    metadata.setdefault("source", "doc.md")
    metadata.setdefault("document_title", "Doc")
    return Document(text=text, metadata=metadata)


class _RecordingRetriever:
    """Stand-in for RetrievalCore that records which method answer_query calls."""

    def __init__(self):
        self.chunks: list = []
        self.calls: list[tuple[str, str, dict | None]] = []

    def keyword_search(self, query, metadata_filter=None):
        self.calls.append(("keyword", query, metadata_filter))
        return []

    def embedding_search(self, query, metadata_filter=None):
        self.calls.append(("embedding", query, metadata_filter))
        return []

    def hybrid_search(self, query, metadata_filter=None):
        self.calls.append(("hybrid", query, metadata_filter))
        return []


# --------------------------------------------------------------------------- #
# __init__  (storage backend selection + config threading)
# --------------------------------------------------------------------------- #
@suite.case("__init__", "builds an in-memory DBManager by default")
def _():
    rag = _rag()
    assert isinstance(rag.db, DBManager), "should construct a DBManager when none is given"


@suite.case("__init__", "reuses an injected DBManager")
def _():
    db = DBManager()
    rag = _rag(db=db)
    assert rag.db is db, "an explicitly passed db should be used as-is"


@suite.case("__init__", "builds a DBManager from db_path")
def _():
    with tempfile.TemporaryDirectory() as tmp:
        rag = _rag(db_path=tmp)
        assert isinstance(rag.db, DBManager)
        # Persisting on disk should leave files behind after an ingest.
        rag.ingest_document(_doc("alpha beta gamma delta"))
        assert any(Path(tmp).iterdir()), "db_path backend should write to disk"


@suite.case("__init__", "rejects passing both db and db_path")
def _():
    try:
        _rag(db=DBManager(), db_path="./somewhere")
        assert False, "expected ValueError when both db and db_path are given"
    except ValueError:
        pass


@suite.case("__init__", "threads retrieval and chunking config into the sub-cores")
def _():
    rag = _rag(top_k=3, chunk_size=8, overlap=2, search_type=SearchType.KEYWORD, system_prompt="SYS")
    assert rag.retrieval_core.top_k == 3, "top_k should reach RetrievalCore"
    assert rag.ingestion_core.chunk_size == 8 and rag.ingestion_core.overlap == 2, "chunking config should reach IngestionCore"
    assert rag.search_type == SearchType.KEYWORD and rag.system_prompt == "SYS"


# --------------------------------------------------------------------------- #
# ingest_document / ingest_path  (hydration of the retriever)
# --------------------------------------------------------------------------- #
@suite.case("ingest_document", "hydrates the retriever with the stored chunks")
def _():
    rag = _rag(chunk_size=5, overlap=1)
    rag.ingest_document(_doc(" ".join(f"word{i}" for i in range(20))))
    assert len(rag.retrieval_core.chunks) > 1, "retriever should be populated after ingestion"


@suite.case("ingest_path", "loads a directory and makes it queryable")
def _():
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "a.md").write_text("# A\n\nalpha beta gamma", encoding="utf-8")
        (Path(tmp) / "b.md").write_text("# B\n\ndelta epsilon zeta", encoding="utf-8")
        rag = _rag()
        rag.ingest_path(tmp)
    assert len(rag.retrieval_core.chunks) == 2, "one chunk per short doc should be searchable"


# --------------------------------------------------------------------------- #
# answer_query  (dispatch + prompt assembly)
# --------------------------------------------------------------------------- #
@suite.case("answer_query", "dispatches to the strategy passed per call")
def _():
    rag = _rag()
    rec = _RecordingRetriever()
    rag.retrieval_core = rec
    rag.answer_query("q", search_type=SearchType.EMBEDDING)
    assert rec.calls[-1][0] == "embedding", f"expected embedding_search, got {rec.calls}"


@suite.case("answer_query", "falls back to the constructor's default search_type")
def _():
    rag = _rag(search_type=SearchType.KEYWORD)
    rec = _RecordingRetriever()
    rag.retrieval_core = rec
    rag.answer_query("q")  # no search_type -> use the configured default
    assert rec.calls[-1][0] == "keyword", f"expected keyword_search, got {rec.calls}"


@suite.case("answer_query", "passes metadata_filter through to the retriever")
def _():
    rag = _rag(search_type=SearchType.HYBRID)
    rec = _RecordingRetriever()
    rag.retrieval_core = rec
    rag.answer_query("q", metadata_filter={"source": "a.md"})
    assert rec.calls[-1][2] == {"source": "a.md"}, f"filter not forwarded: {rec.calls}"


@suite.case("answer_query", "rejects an unknown search type")
def _():
    rag = _rag()
    try:
        rag.answer_query("q", search_type="banana")  # not a SearchType member
        assert False, "expected ValueError for an unknown search type"
    except ValueError:
        pass


@suite.case("answer_query", "stuffs retrieved chunk text into the prompt context")
def _():
    rag = _rag()
    rag.ingest_document(_doc("the quick brown fox jumps over the lazy dog"))
    rag.answer_query("fox", search_type=SearchType.KEYWORD)
    sent = rag.llm.last_prompt  # MockLLM records what it received
    assert "quick brown fox" in sent, "retrieved chunk text should appear in the context"
    assert "Question: fox" in sent, "the question should be appended after the context"


@suite.case("answer_query", "sends the configured system prompt to the LLM")
def _():
    rag = _rag(system_prompt="ONLY use the context.")
    rag.ingest_document(_doc("alpha beta gamma"))
    rag.answer_query("alpha", search_type=SearchType.KEYWORD)
    assert rag.llm.last_system_prompt == "ONLY use the context.", rag.llm.last_system_prompt


# --------------------------------------------------------------------------- #
# retrieve_and_answer  (answer + sources, for the frontend)
# --------------------------------------------------------------------------- #
@suite.case("retrieve_and_answer", "returns the answer and the source chunks")
def _():
    rag = _rag()
    rag.ingest_document(_doc("the quick brown fox jumps over the lazy dog"))
    answer, sources = rag.retrieve_and_answer("fox", search_type=SearchType.KEYWORD)
    assert isinstance(answer, str) and answer, "answer should be a non-empty string"
    assert sources, "the retrieved source chunks should be returned alongside the answer"
    chunk, score = sources[0]
    assert "fox" in chunk.text and isinstance(score, float), (chunk.text, score)


# --------------------------------------------------------------------------- #
# __init__ / _refresh_retrieval  (hydration from a shared store)
# --------------------------------------------------------------------------- #
@suite.case("__init__", "hydrates from an already-populated shared db")
def _():
    db = DBManager()
    # Ingest through one RAGCore (small chunks -> several stored chunks)...
    _rag(db=db, chunk_size=5, overlap=1).ingest_document(
        _doc(" ".join(f"word{i}" for i in range(20)), source="s.md")
    )
    # ...a second RAGCore over the SAME db should see those chunks immediately,
    # before any ingest of its own (this is what each web request relies on).
    fresh = _rag(db=db)
    assert len(fresh.retrieval_core.chunks) > 1, "a new RAGCore should hydrate from the shared store"


@suite.case("_refresh_retrieval", "is a no-op before the first ingest (no collection yet)")
def _():
    rag = _rag()  # nothing ingested -> collection doesn't exist
    assert rag.retrieval_core.chunks == [], "should hydrate to empty without raising"


if __name__ == "__main__":
    raise SystemExit(0 if suite.run() else 1)
