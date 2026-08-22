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

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from openai import AuthenticationError, OpenAIError
from pydantic import BaseModel

import solutions
from clients.embedder import TextEmbedder
from clients.llm import LLMClient
from db_manager import DBManager
from loaders import SUPPORTED_SUFFIXES, document_from_bytes
from rag_core import RAGCore
from retrieval_core import SearchType

# Bind the participant's exported notebook solutions (solutions/part1_ingestion.py,
# solutions/part2_retrieval.py) onto the pipeline. The repo ships no implementation
# of those eight functions, so on a fresh checkout nothing is bound and any query
# raises NotImplementedError.
#
# Unlike the CLI, the web app must not die when the export is incomplete: apply()
# patches whatever IS implemented and then raises. We swallow that raise so the
# server still boots — the /status dashboard then shows exactly which functions
# are still missing, which is far more useful than a failed startup.
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

# Local single-user setup, so a wildcard CORS policy is fine — it just lets the
# Vite dev server (a different origin/port) call this API from the browser.
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
def get_rag(x_openrouter_key: str | None = Header(default=None)) -> RAGCore:
    """Build a RAGCore for this request from the caller-supplied key.

    The key never leaves the request: it constructs the AI clients here and is
    not stored. The shared persistent ``_db`` is injected so the fresh RAGCore
    sees everything already ingested.
    """
    if not x_openrouter_key:
        raise HTTPException(status_code=401, detail="Missing X-OpenRouter-Key header")
    embedder = TextEmbedder(api_key=x_openrouter_key)
    llm = LLMClient(api_key=x_openrouter_key)
    return RAGCore(embedder, llm, db=store(), collection_name=COLLECTION)


def _document_summaries() -> list[dict]:
    """Group stored chunks by source into one entry per document."""
    db = store()
    if not db.collection_exists(COLLECTION):
        return []
    summaries: dict[str, dict] = {}
    for chunk in db.get_all_chunks(COLLECTION):
        source = chunk.metadata.get("source", "unknown")
        entry = summaries.setdefault(
            source,
            {"source": source, "title": chunk.metadata.get("document_title", source), "chunks": 0},
        )
        entry["chunks"] += 1
    return list(summaries.values())


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
        return {"documents": _document_summaries()}


@app.delete("/documents")
def clear_documents() -> dict:
    """Drop everything ingested so far (clean slate)."""
    with _lock:
        store().reset_collection(COLLECTION)
    return {"status": "cleared"}


@app.post("/ingest")
def ingest(
    files: list[UploadFile] = File(...),
    rag: RAGCore = Depends(get_rag),
) -> dict:
    """Upload one or more ``.txt``/``.md`` files and ingest them (load->chunk->embed->store)."""
    ingested: list[dict] = []
    with _lock:
        for upload in files:
            suffix = Path(upload.filename or "").suffix.lower()
            if suffix not in SUPPORTED_SUFFIXES:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file '{upload.filename}'. Supported: {sorted(SUPPORTED_SUFFIXES)}",
                )
            raw = upload.file.read()
            try:
                document = document_from_bytes(upload.filename, raw)
            except ValueError as exc:
                # Unsupported type, non-UTF-8 text, or an unreadable PDF.
                raise HTTPException(status_code=400, detail=str(exc))
            try:
                chunks = rag.ingest_document(document)
            except AuthenticationError:
                raise HTTPException(status_code=401, detail="Invalid OpenRouter API key")
            except OpenAIError as exc:
                raise HTTPException(status_code=502, detail=f"Embedding provider error: {exc}")

            ingested.append(
                {
                    "source": document.metadata["source"],
                    "title": document.metadata["document_title"],
                    "chunks": len(chunks),
                }
            )
    return {"ingested": ingested, "documents": _document_summaries()}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest, rag: RAGCore = Depends(get_rag)) -> QueryResponse:
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty")
    # An empty store and "nothing matched your question" both end in an empty
    # retrieval, but they are different problems with different fixes, and the
    # pipeline's abstention answer ("nothing relevant in the indexed documents")
    # reads as the second when it is really the first. Say so plainly here —
    # this is a UI affordance, not pipeline behaviour, so it stays out of RAGCore.
    if not rag.retrieval_core.chunks:
        raise HTTPException(
            status_code=400,
            detail="No documents have been ingested yet — upload a document before asking a question.",
        )
    with _lock:
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
