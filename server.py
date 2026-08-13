"""FastAPI backend that puts the RAG pipeline behind an HTTP API.

Designed for the course setup: **each participant runs this locally** and
**supplies their own OpenRouter API key**, which the frontend sends on every
request in the ``X-OpenRouter-Key`` header. The key is used to build the AI
clients for that one request and is never written to disk or logged.

Run it:

    uv run uvicorn server:app --reload

The store is a single persistent Qdrant (``./qdrant_data`` by default) shared
for the life of the process, so uploaded documents survive across requests and
restarts. A fresh ``RAGCore`` is built per request over that shared store — the
injected ``db`` plus hydrate-on-init mean it immediately sees everything already
ingested.
"""

from __future__ import annotations

import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from openai import AuthenticationError, OpenAIError
from pydantic import BaseModel

import solutions
from clients.embedder import TextEmbedder
from clients.llm import LLMClient
from db_manager import DBManager
from loaders import document_from_bytes
from rag_core import RAGCore
from models import SearchType

# If a participant has exported their notebook solutions into solutions/, run the
# app on *their* implementations. A no-op on a fresh checkout (stubs are skipped),
# so the reference implementation keeps the app working out of the box.
# Set RAG_IMPL=reference|student to force one implementation (default "auto").
#
# Unlike the CLI, the web app must not die when RAG_IMPL=student is incomplete:
# apply() patches whatever IS implemented (the rest fall back to reference) and
# then raises. We swallow that raise so the server still boots — the /status
# dashboard then shows exactly which functions are still missing.
try:
    solutions.apply()
except RuntimeError:
    pass

DB_PATH = os.environ.get("RAG_DB_PATH", "./qdrant_data")
COLLECTION = os.environ.get("RAG_COLLECTION", "documents")

# The store is opened in the lifespan (startup), NOT at import time. On-disk
# Qdrant is single-writer; importing this module happens in more than one
# process under `uvicorn --reload` (reloader + worker), so opening at import
# would make them fight over the storage lock. Lifespan runs only in the
# serving worker, so the store is opened exactly once.
_db: DBManager | None = None

# Local on-disk Qdrant isn't built for concurrent access, and FastAPI runs sync
# handlers in a threadpool, so a single lock serializes store access. Fine at
# course scale / single user.
_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _db
    _db = DBManager(path=DB_PATH)
    try:
        yield
    finally:
        _db.close()
        _db = None


def store() -> DBManager:
    """The process-wide store, or 503 if accessed before startup completed."""
    if _db is None:
        raise HTTPException(status_code=503, detail="Store is not ready")
    return _db


app = FastAPI(title="RAG Course API", version="0.1.0", lifespan=lifespan)

# Local single-user / workshop setup — a wildcard CORS policy lets the Vite dev
# server (a different origin/port) and Colab proxy origins call this API from
# the browser without preflight failures.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# Request / response models
# --------------------------------------------------------------------------- #
class QueryRequest(BaseModel):
    query: str
    search_type: SearchType | None = None  # None -> server/orchestrator default
    metadata_filter: dict | None = None


class Source(BaseModel):
    text: str
    source: str | None = None
    document_title: str | None = None
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]


# --------------------------------------------------------------------------- #
# Per-request RAGCore built from the caller's key
# --------------------------------------------------------------------------- #
def get_rag(api_key: str | None) -> RAGCore:
    """Build a RAGCore from the caller-supplied OpenRouter key.

    Called manually inside the ``_lock`` block of each endpoint (not via
    ``Depends``), so that ``RAGCore.__init__`` → ``_refresh_retrieval`` →
    ``db.get_all_chunks`` runs under the same lock as the operation that
    follows it. This prevents a concurrent ingest from slipping in between
    hydration and the actual query or ingest step.

    **Per-request construction is intentional at workshop scale**: a fresh
    RAGCore is built for every request, which triggers a full DB read each
    time. For a single-user local setup this is acceptable. A production
    service would use a singleton RAGCore (or an LRU cache keyed on the API
    key) so the DB scan happens once, not per request.

    The key never leaves the request: it constructs the AI clients here and
    is not stored anywhere.
    """
    if not api_key:
        raise HTTPException(status_code=401, detail="Missing X-OpenRouter-Key header")
    embedder = TextEmbedder(api_key=api_key)
    llm = LLMClient(api_key=api_key)
    return RAGCore(embedder, llm, db=store(), collection_name=COLLECTION)


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/status")
def implementation_status() -> dict:
    """Which notebook-exported functions are implemented — drives the progress dashboard."""
    return solutions.status()


@app.get("/search-types")
def search_types() -> dict:
    """The retrieval strategies the UI can offer (populates the dropdown)."""
    return {"search_types": [s.value for s in SearchType]}


@app.get("/documents")
def list_documents() -> dict:
    """The current corpus — readable without a key (it's the caller's own store)."""
    with _lock:
        return {"documents": store().list_documents(COLLECTION)}


@app.delete("/documents")
def clear_documents() -> dict:
    """Drop everything ingested so far (clean slate)."""
    with _lock:
        store().reset_collection(COLLECTION)
    return {"status": "cleared"}


@app.post("/ingest")
def ingest(
    files: list[UploadFile] = File(...),
    x_openrouter_key: str | None = Header(default=None),
) -> dict:
    """Upload one or more ``.txt``/``.md`` files and ingest them (load->chunk->embed->store)."""
    ingested: list[dict] = []
    with _lock:
        rag = get_rag(x_openrouter_key)
        for upload in files:
            raw = upload.file.read()
            try:
                document = document_from_bytes(upload.filename, raw)
            except ValueError as exc:
                # Unsupported type (document_from_bytes lists supported extensions),
                # non-UTF-8 text, or an unreadable PDF.
                raise HTTPException(status_code=400, detail=str(exc))
            try:
                chunks = rag.ingest_document(document)
            except AuthenticationError:
                raise HTTPException(status_code=401, detail="Invalid OpenRouter API key")
            except OpenAIError as exc:
                raise HTTPException(status_code=502, detail=f"Embedding provider error: {exc}")

            ingested.append(
                {
                    "source": document.metadata.get("source", "unknown"),
                    "title": document.metadata.get("document_title", "unknown"),
                    "chunks": len(chunks),
                }
            )
    return {"ingested": ingested, "documents": store().list_documents(COLLECTION)}


@app.post("/query", response_model=QueryResponse)
def query(
    req: QueryRequest,
    x_openrouter_key: str | None = Header(default=None),
) -> QueryResponse:
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty")
    with _lock:
        rag = get_rag(x_openrouter_key)
        try:
            answer, results = rag.retrieve_and_answer(
                req.query,
                search_type=req.search_type,
                metadata_filter=req.metadata_filter,
            )
        except AuthenticationError:
            raise HTTPException(status_code=401, detail="Invalid OpenRouter API key")
        except OpenAIError as exc:
            raise HTTPException(status_code=502, detail=f"LLM provider error: {exc}")

    sources = [
        Source(
            text=chunk.text,
            source=chunk.metadata.get("source"),
            document_title=chunk.metadata.get("document_title"),
            score=score,
        )
        for chunk, score in results
    ]
    return QueryResponse(answer=answer, sources=sources)


# --------------------------------------------------------------------------- #
# Static file serving — serves the React frontend build from the same port.
#
# This MUST be after all API route definitions. FastAPI's mount() is greedy:
# it catches any path not already matched by an explicit route. Placing it
# here means /health, /query, /documents etc. are matched first, and
# everything else (index.html, JS, CSS, images) falls through to the static
# file server.
#
# Why same-port serving matters for Colab:
# Colab's proxy requires browser session cookies. If the frontend and backend
# are on different ports (different proxy URLs), cross-origin fetch calls
# won't carry the right cookies and Colab's proxy returns 404 before the
# request reaches FastAPI. Serving everything from one port = one proxy URL =
# one origin = no CORS issues.
# --------------------------------------------------------------------------- #
_frontend_dist = Path(__file__).resolve().parent / "frontend" / "dist"
print(f"[server] frontend dist path: {_frontend_dist}")
print(f"[server] exists: {_frontend_dist.is_dir()}")
if _frontend_dist.is_dir():
    from fastapi.staticfiles import StaticFiles
    app.mount("/", StaticFiles(directory=str(_frontend_dist), html=True))
    print("[server] Static files mounted at /")
else:
    print("[server] ⚠️  dist/ not found — frontend not served")
