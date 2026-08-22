# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

This project is the `project/` folder of the public course repo
[eth-fdd-fs26/FDD-WE5-public](https://github.com/eth-fdd-fs26/FDD-WE5-public) (branch `main`).
**Every command and path below is relative to `project/`**, which is also the working directory the
notebook `cd`s into. Nothing in this project reads or writes outside that folder.

## Commands

This is a Python 3.12+ project managed with **UV** (not pip/poetry). There is no build step and no configured linter/formatter — do not invent one.

```bash
uv sync                          # install/sync env from uv.lock
uv add <package>                 # REQUIRED whenever a new third-party import is introduced
                                 # (keeps pyproject.toml + uv.lock in sync; never hand-edit deps)

# Run the end-to-end demo (ingests data/, answers a sample query)
OPENROUTER_API_KEY=sk-... uv run python main.py

# Tests — a custom harness, NOT pytest. Run all suites (combined table, CI exit code):
uv run python -m tests

# Run one suite (the finest granularity — there is no per-case filter):
uv run python -m tests.test_retrieval_core
uv run python -m tests.test_ingestion_core
uv run python -m tests.test_db_manager
uv run python -m tests.test_rag_core
uv run python -m tests.test_evaluation
uv run python -m tests.test_server
```

Tests run fully **offline** via `tests/mocks.py` and an in-memory Qdrant — no API key or network needed.
On a fresh checkout the suite is **red on purpose**: the eight participant functions have no
implementation in the repo, so every check touching one fails with a `NotImplementedError` naming its
notebook exercise. `tests/__init__.py` calls `solutions.apply(verbose=False)` at import (swallowing the
"not everything exported" `RuntimeError`), so the suite goes green as a participant exports their work.
Only `test_db_manager` and `test_evaluation` are independent of the exercises and always pass.

Two embedder doubles, and the difference matters: `MockEmbedder` returns one **fixed** vector (deterministic cosine scores — right for unit tests, but every chunk ties, so it must never back a demo or a measurement), while `HashingEmbedder` hashes words into a normalised vector so rankings are real. `HashingEmbedder` is what the notebook's demos and the whole evaluation section run on; it scores *word overlap*, not meaning, so its semantic numbers are a floor, not an estimate of dense retrieval. Only `main.py` and the web backend need a real OpenRouter key.

### Course notebook (the student deliverable)

`notebook/compliance_rag_project.ipynb` is the only notebook in this **public** repo, and it is a *generated* artifact. Its generator (`build_notebook.py`, nbformat) and the answer-key notebook it can emit both contain all eight solutions verbatim, so **neither is distributed here** — they live in the private instructor repo, which is where the notebook is regenerated and from where the student `.ipynb` is copied over. Do not add either to this repo, and do not reconstruct the solutions into it.

Students implement functions in the notebook and **monkey-patch** them onto the real modules; export cells write `solutions/part1_ingestion.py` and `solutions/part2_retrieval.py`.

A *stub* is not an empty body — it is the reference solution with **holes punched in it**: the scaffolding (loops, guard clauses, order of operations) is given, and the expressions that carry the idea are replaced by the marker `TODO`, each with a one-line `# ✏️` hint. `TODO` is a live sentinel object defined in a collapsed §0.3 cell (`_Blank`), whose every dunder raises `NotImplementedError` with a fill-me-in message — so a blank left unfilled fails loudly at the test cell instead of passing as a wrong answer. Consequences worth knowing: **(a)** solution bodies must never contain the substring `TODO`, and **(b)** the export cells call `has_blanks(fn)` (source contains `TODO`) and skip such functions, so a half-finished notebook exports a partial file rather than one that crashes the app on import.

Participants implement **8 functions**: `sliding_window`, `ingest_document`, `apply_metadata_filter` (Part 1) and `cosine_similarity`, `keyword_search`, `reciprocal_rank_fusion`, `hybrid_search`, `retrieve_and_answer` (Part 2).

**These eight have no implementation in the repo.** Their bodies in `chunking.py` / `ingestion_core.py` / `retrieval_core.py` / `rag_core.py` are a docstring stating the contract plus a call to `solutions.not_implemented("<name>")`, which raises a `NotImplementedError` naming the notebook exercise and the export file. No copy of the real bodies is distributed with this repo, so a participant cannot run the app on borrowed code. Deleting a body is therefore a *deliberate* state; do not "helpfully" restore one.

`solutions/__init__.py` is the single source of truth: `PART1_SPECS`/`PART2_SPECS` (each `PatchSpec` carries the notebook `section`), `apply()`, `status()`, and `not_implemented()`. `apply()` binds whatever the two exported files provide and raises `RuntimeError` listing anything still missing — there is no fallback. `main.py` catches it and exits 1; `server.py` swallows it so the process still boots and `GET /status` can show which functions are missing (the frontend's Implementation panel); `tests/__init__.py` swallows it too. There is no `RAG_IMPL` / `--impl` mode switch — it was removed along with the reference implementations. The notebook patches the *same* targets explicitly, cell by cell (transparent for teaching), so the two paths stay in sync.

Because the repo's `keyword_search` raises, every notebook demo must come **after** the cell that patches what it uses. The §2.0 "watch BM25 fail" demo runs before §2.2, so it builds its own throwaway `BM25Okapi` ranking inline rather than calling `RetrievalCore.keyword_search`.

### Web app (FastAPI backend + React frontend)

```bash
# Backend (serves the RAG pipeline over HTTP on :8000)
uv run uvicorn server:app --reload

# Frontend (Vite dev server on :5173; talks to the backend at http://localhost:8000)
cd frontend && npm install && npm run dev
```

The backend reads no key from the environment — **each request carries the user's OpenRouter key** in the `X-OpenRouter-Key` header. Override the store location with `RAG_DB_PATH` (default `./qdrant_data`, gitignored) or the backend's expected origin in the frontend with `VITE_API_BASE`.

## Architecture

A teaching-oriented RAG system. Both AI clients (`clients/embedder.py`, `clients/llm.py`) are thin wrappers over the OpenAI SDK pointed at **OpenRouter** (`https://openrouter.ai/api/v1`); they differ only in target model.

The system has a **write side** and a **read side**, and the key insight is how they connect:

- **Write (ingestion):** `IngestionCore` runs `load → chunk → embed → store`. It loads `.txt`/`.md`/`.pdf` into `Document`s (`loaders.py`, which is where metadata `source`/`document_title` is attached; PDFs have their text layer extracted with `pypdf` — scanned/image-only PDFs yield no text), splits via `chunking.py`, batch-embeds, and persists into **Qdrant** through `DBManager` (`db_manager.py`).
- **Measurement (`evaluation.py`):** a 12-question gold set over `data/` (6 `lexical` + 6 `semantic`), plus `recall_at_k` (gold **document** in top k), `answer_recall_at_k` (the gold **passage** text is really in the retrieved context) and `context_cost` (words handed to the LLM). All three exist because document recall alone cannot distinguish chunking strategies — an un-chunked corpus "hits" every time at ~10× the context. `tests/test_evaluation.py` asserts every `gold_source` and `answer_phrase` still matches `data/`, so the set can't silently rot.
- **Read (retrieval):** `RetrievalCore` does **not** query Qdrant. It searches an **in-memory `self.chunks` list** — the "simple vector database" — with three strategies: `keyword_search` (BM25, rebuilt per call), `embedding_search` (numpy cosine), and `hybrid_search` (the two fused via Reciprocal Rank Fusion, `rrf_k=60`, which sidesteps BM25-vs-cosine scale mismatch). Hybrid **over-retrieves**: each branch returns `top_k * overretrieve` candidates (default 20) so fusion has a real pool to rerank — fusion can only reorder what it is given, and it is nearly free here because both branches already score every chunk. `overretrieve=1` is the old, narrow behaviour (measured: recall 0.75 → 0.83, plateauing past ~5×). `SearchType` selects the strategy. `RetrievalCore.chunks` is a **property**: its setter invalidates the cached BM25 index (`_bm25_index`), which is cached only for the unfiltered corpus — a metadata filter yields different IDF and is built ad hoc.
- **Not implemented here:** `sliding_window`, `IngestionCore.ingest_document`, and `RetrievalCore`'s `keyword_search` / `_cosine_similarity` / `hybrid_search` / `_apply_metadata_filter` / `_reciprocal_rank_fusion`, plus `RAGCore.retrieve_and_answer`, all raise until a participant's `solutions/` files are bound. `embedding_search` is provided but depends on two of them, so it raises too.
- **Abstention:** `RAGCore.min_score` (default `None`) drops results below a threshold in `_retrieve`; when nothing survives, `retrieve_and_answer` returns `NO_CONTEXT_ANSWER` **without calling the LLM**. The right threshold is strategy-specific (cosine ∈ [-1,1], BM25 unbounded, RRF ≈ 1/60) — and fragile: with no stopword removal, a nonsense question phrased as a full sentence still scores ~7.5 on BM25 here.
- **The bridge:** `RAGCore` (`rag_core.py`) is the orchestrator. After every ingest it calls `_refresh_retrieval()`, which reloads `RetrievalCore.chunks` from Qdrant via `DBManager.get_all_chunks`. This reload is the *only* thing that makes newly written chunks searchable. `receive_query` then retrieves → stuffs chunk texts into a context block → asks the LLM to answer using only that context.

Data models (`models/`) are UUID-id dataclasses that reference each other **by ID, not object** (serializable, no cycles): `Document.chunk_ids` ↔ `Chunk.document_id`. `Chunk.embedding` is `list[float] | None` (kept as a list, not numpy, to stay JSON-serializable).

### Storage backend selection (set once on `RAGCore`/`DBManager`)
- `db=` → reuse an injected `DBManager`; `db_path=` → on-disk persistent Qdrant; neither → in-memory ephemeral. Passing both `db` and `db_path` raises `ValueError`.
- Chunking/retrieval config (`chunk_size`, `overlap`, `strategy`, `top_k`, `search_type`, `system_prompt`) is set on `RAGCore` and threaded into the sub-cores; per-method args still override.

### Web layer (`server.py` + `frontend/`)
A thin HTTP shell over the same `RAGCore` — the pipeline itself is unaware of it.
- **One shared persistent `DBManager`** lives for the process lifetime; a **fresh `RAGCore` is built per request** over it (injected `db=` + hydrate-on-init mean it immediately sees everything already ingested). This is how the per-request key model works without rebuilding the store.
- The OpenRouter key arrives per request (`X-OpenRouter-Key` header → `get_rag` dependency builds the clients) and is never persisted or logged.
- Local on-disk Qdrant isn't concurrency-safe, so a module-level `threading.Lock` serializes store access (fine at single-user / course scale).
- `POST /query` returns **400** when the store is empty ("no documents ingested yet"): an empty store and "nothing matched" both end in an empty retrieval, but they need different fixes, and the pipeline's abstention answer reads as the second. That distinction is a UI affordance, so it lives in `server.py`, not `RAGCore`.
- `tests/test_server.py` covers the API offline (`TestClient` + `HashingEmbedder`/`MockLLM`, `get_rag` overridden). It points `server.DB_PATH` at a fresh `tempfile.mkdtemp()` per case — local Qdrant takes an exclusive folder lock, and a test must never touch the developer's real `./qdrant_data`.
- Endpoints: `POST /ingest` (multipart `.txt`/`.md`/`.pdf` upload → `document_from_bytes` → `ingest_document`, no temp files; bad type / non-UTF-8 / unreadable PDF → HTTP 400), `POST /query` (→ `retrieve_and_answer`, returns answer **plus source chunks** for citations), `GET/DELETE /documents`, `GET /search-types`, `GET /health`. OpenRouter auth failures are mapped to HTTP 401.
- `frontend/` is a Vite + React app: key field (sessionStorage only), upload, query box with a strategy selector, document list, and an answer view with expandable sources.

### Metadata flow & validation
Metadata enters at `loaders.py` (`source`, `document_title`), rides on each `Chunk`, and is flattened into the Qdrant payload. `metadata_schema.py::validate_payload` enforces the contract. The active required set is deliberately **minimal** (`text`, `source`, `document_title`) — `document_type`, `jurisdiction`, `year`, etc. are defined but commented out. Retrieval can narrow by any metadata key via `metadata_filter` (`{key: value}`, AND-matched before scoring).

## Gotchas

- The notebook's §0.1 setup cell clones `FDD-WE5-public` (public, `main`, `--depth 1`, no token) and `cd`s into `project/`. Locally it walks *up* from the launch directory until it finds `chunking.py`, so it works from `project/` or `project/notebook/`. If the project ever moves, that cell (`REPO_URL` / `REPO_BRANCH` / `PROJECT_SUBDIR`) is what needs changing — in the generator, in the instructor repo.

- `DBManager.get_all_chunks` returns **Qdrant's internal order, not insertion order**; `_refresh_retrieval` re-sorts by `(source, document_id, index)`, so `RetrievalCore.chunks` is in document order.
- `validate_payload` now runs on **both** write paths (`insert_points` and `insert_chunks`), so a chunk that lost its metadata fails at ingestion instead of degrading retrieval silently.
- Qdrant is a **persistence layer only** — all live ranking is the in-memory path in `RetrievalCore`. That is deliberate (seeing BM25/cosine/fusion written out is the point of the course), not an oversight.
- `chunking.sliding_window` must reject `overlap >= chunk_size` (a window that never advances). `strategy="whole_document"` is the deliberate "no chunking" baseline and *is* implemented in the repo — it is not one of the eight exercises.
- Executing the notebook runs its export cells, overwriting `solutions/part1_ingestion.py` / `part2_retrieval.py`. Restore the shipped `@todo` placeholders afterwards, or the suite looks green when a fresh checkout should be red.

## Testing conventions

The harness (`tests/harness.py`, `TestSuite`) groups checks by the **function** they exercise via `@suite.case("function_name", "description")`, so the `rich` report points at the failing function. `tests/__main__.py` runs each suite as a subprocess. `test_db_manager.py` is the exception — a self-contained script with plain `assert`s and its own runner (no `rich`, no harness).
