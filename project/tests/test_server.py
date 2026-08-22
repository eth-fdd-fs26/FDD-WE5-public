"""Tests for server.py — the HTTP shell around the pipeline.

These exercise *wiring and contracts*, not the RAG algorithms (those have their
own suites): status codes, error mapping, and the fact that the key never leaves
the request. Everything runs offline — ``get_rag`` is overridden with a RAGCore
built on ``HashingEmbedder`` / ``MockLLM``, so no key and no network are needed.

Run it:

    uv run python -m tests.test_server
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient

import server
from rag_core import RAGCore
from tests.harness import TestSuite
from tests.mocks import HashingEmbedder, MockLLM

suite = TestSuite("Server — Test Results")

KEY_HEADER = {"X-OpenRouter-Key": "sk-or-v1-offline-test-key"}


#: Temp store directories to remove on teardown.
_TEMP_DIRS: list[str] = []


def _client(ingest: str | None = None) -> TestClient:
    """A TestClient whose RAGCore is offline, over a private on-disk store.

    Each call gets its own temp directory: local Qdrant takes an exclusive lock
    on its storage folder, and the developer's real ``./qdrant_data`` (the
    server default) must never be touched by a test run.
    """
    store_dir = tempfile.mkdtemp(prefix="rag-test-store-")
    _TEMP_DIRS.append(store_dir)
    server.DB_PATH = store_dir  # read by the lifespan handler below

    client = TestClient(server.app)
    client.__enter__()  # run lifespan: builds the shared _db at DB_PATH
    server.app.dependency_overrides[server.get_rag] = lambda: RAGCore(
        HashingEmbedder(), MockLLM(), db=server.store(), collection_name=server.COLLECTION
    )
    if ingest is not None:
        rag = RAGCore(HashingEmbedder(), MockLLM(), db=server.store(), collection_name=server.COLLECTION)
        rag.ingest_path(ingest)
    return client


def _close(client: TestClient) -> None:
    server.app.dependency_overrides.clear()
    client.__exit__(None, None, None)  # lifespan closes the store, releasing its lock


# --------------------------------------------------------------------------- #
# /health, /search-types, /status
# --------------------------------------------------------------------------- #
@suite.case("health", "reports ok")
def _():
    client = _client()
    try:
        assert client.get("/health").json() == {"status": "ok"}
    finally:
        _close(client)


@suite.case("search_types", "exposes exactly the SearchType values the frontend offers")
def _():
    client = _client()
    try:
        assert client.get("/search-types").json()["search_types"] == ["keyword", "embedding", "hybrid"]
    finally:
        _close(client)


@suite.case("implementation_status", "reports every patch target so the dashboard can render it")
def _():
    client = _client()
    try:
        body = client.get("/status").json()
        assert body["total"] == 8, f"expected 8 patch targets, got {body['total']}"
        assert {"name", "target", "implemented", "reason"} <= set(body["functions"][0])
    finally:
        _close(client)


# --------------------------------------------------------------------------- #
# get_rag  (auth)
# --------------------------------------------------------------------------- #
@suite.case("get_rag", "a missing key is rejected with 401")
def _():
    client = _client()
    try:
        # No dependency override for this one: exercise the real get_rag.
        server.app.dependency_overrides.clear()
        response = client.post("/query", json={"query": "anything"})
        assert response.status_code == 401, f"expected 401, got {response.status_code}"
    finally:
        _close(client)


# --------------------------------------------------------------------------- #
# /query
# --------------------------------------------------------------------------- #
@suite.case("query", "an empty query is rejected with 400")
def _():
    client = _client()
    try:
        assert client.post("/query", json={"query": "   "}, headers=KEY_HEADER).status_code == 400
    finally:
        _close(client)


@suite.case("query", "an empty store says so, instead of 'nothing relevant'")
def _():
    # These are different problems with different fixes; the abstention answer
    # would send the user looking for a better question instead of uploading.
    client = _client()
    try:
        response = client.post("/query", json={"query": "annual leave"}, headers=KEY_HEADER)
        assert response.status_code == 400, f"expected 400, got {response.status_code}"
        assert "ingested" in response.json()["detail"].lower()
    finally:
        _close(client)


@suite.case("query", "answers over an ingested corpus and returns its sources")
def _():
    client = _client(ingest="data")
    try:
        response = client.post(
            "/query",
            json={"query": "Is MFA mandatory for VPN connections?", "search_type": "keyword"},
            headers=KEY_HEADER,
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["answer"], "an answer should come back"
        assert body["sources"], "the frontend needs sources for citations"
        assert {"text", "source", "document_title", "score"} <= set(body["sources"][0])
    finally:
        _close(client)


@suite.case("query", "metadata_filter narrows the sources to that document")
def _():
    client = _client(ingest="data")
    try:
        body = client.post(
            "/query",
            json={
                "query": "leave",
                "search_type": "keyword",
                "metadata_filter": {"source": "leave_and_absence_policy.md"},
            },
            headers=KEY_HEADER,
        ).json()
        assert body["sources"], "the filter should not empty the results"
        assert all(s["source"] == "leave_and_absence_policy.md" for s in body["sources"])
    finally:
        _close(client)


# --------------------------------------------------------------------------- #
# /ingest
# --------------------------------------------------------------------------- #
@suite.case("ingest", "accepts a .md upload and reports the chunks created")
def _():
    client = _client()
    try:
        response = client.post(
            "/ingest",
            files={"files": ("policy.md", b"# Test Policy\n\nEmployees get 25 days of leave.", "text/markdown")},
            headers=KEY_HEADER,
        )
        assert response.status_code == 200, response.text
        entry = response.json()["ingested"][0]
        assert entry["source"] == "policy.md" and entry["title"] == "Test Policy"
        assert entry["chunks"] >= 1
    finally:
        _close(client)


@suite.case("ingest", "an unsupported file type is rejected with 400")
def _():
    client = _client()
    try:
        response = client.post(
            "/ingest", files={"files": ("virus.exe", b"MZ", "application/octet-stream")}, headers=KEY_HEADER
        )
        assert response.status_code == 400, f"expected 400, got {response.status_code}"
        assert "Unsupported file" in response.json()["detail"]
    finally:
        _close(client)


@suite.case("ingest", "a non-UTF-8 text file is rejected with 400, not a 500")
def _():
    client = _client()
    try:
        response = client.post(
            "/ingest", files={"files": ("bad.txt", b"\xff\xfe\x00bad", "text/plain")}, headers=KEY_HEADER
        )
        assert response.status_code == 400, f"expected 400, got {response.status_code}"
    finally:
        _close(client)


@suite.case("ingest", "an unreadable PDF is rejected with 400, not a 500")
def _():
    client = _client()
    try:
        response = client.post(
            "/ingest", files={"files": ("bad.pdf", b"not a pdf at all", "application/pdf")}, headers=KEY_HEADER
        )
        assert response.status_code == 400, f"expected 400, got {response.status_code}"
        assert "PDF" in response.json()["detail"]
    finally:
        _close(client)


# --------------------------------------------------------------------------- #
# /documents
# --------------------------------------------------------------------------- #
@suite.case("documents", "lists one entry per document with its chunk count")
def _():
    client = _client(ingest="data")
    try:
        documents = client.get("/documents").json()["documents"]
        assert len(documents) == 9, f"expected 9 policies, got {len(documents)}"
        assert all(d["chunks"] > 0 for d in documents)
    finally:
        _close(client)


@suite.case("clear_documents", "empties the store")
def _():
    client = _client(ingest="data")
    try:
        assert client.delete("/documents").json() == {"status": "cleared"}
        assert client.get("/documents").json()["documents"] == []
    finally:
        _close(client)


def _cleanup() -> None:
    for path in _TEMP_DIRS:
        shutil.rmtree(path, ignore_errors=True)


if __name__ == "__main__":
    try:
        ok = suite.run()
    finally:
        _cleanup()
    raise SystemExit(0 if ok else 1)
