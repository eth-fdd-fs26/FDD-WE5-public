# Tests

Self-checking tests for the parts you implement. Each suite groups its checks by
the **function** they exercise and prints a colored report, so you can see at a
glance which functions pass and which still need work.

## Running

```bash
# Run EVERY suite at once (combined pass/fail summary; non-zero exit on failure)
uv run python -m tests

# ...or run an individual suite:

# Retrieval core (keyword / embedding / hybrid search and their helpers)
uv run python -m tests.test_retrieval_core

# Ingestion (chunking, loaders, and the load -> chunk -> embed -> store pipeline)
uv run python -m tests.test_ingestion_core

# RAG orchestrator (storage selection, retriever hydration, query dispatch)
uv run python -m tests.test_rag_core
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
- `test_retrieval_core.py` — the checks for `retrieval_core.py`.
- `test_ingestion_core.py` — the checks for `chunking.py`, `loaders.py`, and
  `ingestion_core.py` (uses an in-memory `DBManager`, so still no network).
- `test_rag_core.py` — the checks for `rag_core.py` (orchestrator wiring:
  storage-backend selection, retriever hydration, query dispatch).

New suites can follow the same pattern and be run the same way.
