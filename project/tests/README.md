# Tests

Self-checking tests for the parts you implement. Each suite groups its checks by
the **function** they exercise and prints a colored report, so you can see at a
glance which functions pass and which still need work.

> **A fresh checkout is red, and that is correct.** The eight functions you build
> in the notebook have no implementation in this repo — every check that touches
> one fails with `NotImplementedError` until you write it and export it into
> `solutions/`. Treat the report as your to-do list: it goes green as you go.

## Running

All commands run from the repo's `project/` folder.

```bash
# Run EVERY suite at once (combined pass/fail summary; non-zero exit on failure)
uv run python -m tests

# ...or run an individual suite (the finest granularity — there is no per-case filter):

# Retrieval core (keyword / embedding / hybrid search and their helpers)
uv run python -m tests.test_retrieval_core

# Ingestion (chunking, loaders, and the load -> chunk -> embed -> store pipeline)
uv run python -m tests.test_ingestion_core

# RAG orchestrator (storage selection, retriever hydration, query dispatch)
uv run python -m tests.test_rag_core

# The vector store wrapper (plain asserts, its own runner — no rich, no harness)
uv run python -m tests.test_db_manager

# The evaluation gold set and its metrics
uv run python -m tests.test_evaluation

# The FastAPI layer (TestClient over a temp store, mock embedder + LLM)
uv run python -m tests.test_server
```

The command exits with code `0` if everything passes, `1` otherwise. The final
panel lists any **functions needing attention** — start there.

These tests run fully offline: they use mock clients (`tests/mocks.py`), so no
API key or network access is required.

## How it works

- `harness.py` — a tiny test runner. Register a check with
  `@suite.case("function_name", "what it checks")` above a function that
  `assert`s the expected behavior. A failed `assert` (or any exception) marks the
  check as failed and shows the message.
- `mocks.py` — `MockEmbedder` / `MockLLM` stand-ins so tests are deterministic.
  `MockEmbedder` returns one *fixed* vector (every chunk ties — right for unit
  tests, useless for ranking); `HashingEmbedder` hashes words into a normalised
  vector, so rankings are real. The notebook's demos and measurements use the
  latter.
- `test_retrieval_core.py` — the checks for `retrieval_core.py`.
- `test_ingestion_core.py` — the checks for `chunking.py`, `loaders.py`, and
  `ingestion_core.py` (uses an in-memory `DBManager`, so still no network).
- `test_rag_core.py` — the checks for `rag_core.py` (orchestrator wiring:
  storage-backend selection, retriever hydration, query dispatch).
- `test_db_manager.py` — the checks for `db_manager.py`. The exception to the
  pattern: a self-contained script with plain `assert`s and its own runner.
- `test_evaluation.py` — the checks for `evaluation.py`, including that every
  `gold_source` and `answer_phrase` in the gold set still matches `data/`.
- `test_server.py` — the checks for `server.py`, over a fresh temp store per case
  (local Qdrant takes an exclusive folder lock; a test must never touch your real
  `./qdrant_data`).

New suites can follow the same pattern and be run the same way.
