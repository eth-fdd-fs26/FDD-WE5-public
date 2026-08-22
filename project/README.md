# Compliance Q&A — a RAG project

A Retrieval-Augmented Generation (RAG) system you build yourself: ask questions about company
policy and get answers grounded in a provided document set.

This is the **`project/`** folder of the [FDD-WE5-public](https://github.com/eth-fdd-fs26/FDD-WE5-public)
course repo. Every path and command below is relative to this folder:

```bash
git clone https://github.com/eth-fdd-fs26/FDD-WE5-public.git
cd FDD-WE5-public/project
```

> **The eight core functions are not implemented here.** Their bodies raise
> `NotImplementedError`. There is no reference version to fall back on, so the test suite is red
> and the web app cannot answer a question until *you* write them in the notebook and export them.
> That is the project.

## Start here: the course notebook

- **`notebook/compliance_rag_project.ipynb`** — the student notebook (exercise stubs + tests).
  Built for **Google Colab**; works in local Jupyter too.

  Open it in Colab straight from GitHub
  ([File → Open notebook → GitHub](https://colab.research.google.com/github/eth-fdd-fs26/FDD-WE5-public/blob/main/project/notebook/compliance_rag_project.ipynb)),
  then run the first cell. The repo is **public**, so the clone needs no token — the setup cell
  clones it, installs the dependencies, and `cd`s into this `project/` folder for you.

  The one *optional* credential is `OPENROUTER_API_KEY`, only for the two live demos at the end of
  §2.5 and §2.6. Add it via Colab's **🔑 Secrets** panel with "Notebook access" ON — never pasted
  into a cell. Locally, `export OPENROUTER_API_KEY=...` before launching Jupyter. **Everything
  else — every exercise and every test — runs fully offline with mock clients.**

The answer key is **not** in this repo — it is held by the course staff. Your own test cells are
how you know a function is right: each exercise is followed by a cell that checks it and prints a
per-function report.

You implement each function in the notebook and *monkey-patch* it onto the real project code, so
the provided scaffolding and tests run against your implementation. At the end of each part an
export cell writes your functions to `solutions/part1_ingestion.py` and
`solutions/part2_retrieval.py`.

**Eight functions in total:**

| Part | Function | What it teaches |
|---|---|---|
| 1.1 | `sliding_window` | chunking, and why overlap exists |
| 1.2 | `ingest_document` | the load → chunk → embed → store pipeline |
| 1.3 | `apply_metadata_filter` | narrowing the search space before scoring |
| 2.1 | `cosine_similarity` | vectorised semantic ranking |
| 2.2 | `keyword_search` | BM25, and why the index is cached |
| 2.3 | `reciprocal_rank_fusion` | fusing rankings on incompatible score scales |
| 2.4 | `hybrid_search` | over-retrieve, then fuse |
| 2.6 | `retrieve_and_answer` | context stuffing — and when to abstain |

Section **2.5 measures all of it**. `evaluation.py` ships a 12-question gold set over `data/`
(half phrased in the documents' own vocabulary, half the way a person actually asks) and three
metrics — document recall@k, answer-passage recall@k, and the context cost of getting it. Every
design decision in the notebook is checked against them rather than asserted, including the ones
where the measurement disagrees with the tidy story.

## Tests

A custom offline harness (not pytest) — no API key and no network needed:

```bash
uv sync                                      # once, to create the environment
uv run python -m tests                       # every suite, non-zero exit on failure
uv run python -m tests.test_ingestion_core   # or one suite at a time
```

On a fresh checkout most of it **fails**, by design: the checks covering the eight functions are
your to-do list. The final panel names the functions still needing attention. See `tests/README.md`.

## Running the full app on your own implementations

Drop your two exported files into `solutions/`. On startup `main.py` / `server.py` call
`solutions.apply()`, which binds your functions onto the live app and reports any still missing.

```bash
# CLI demo: ingests data/, answers one sample question
OPENROUTER_API_KEY=sk-... uv run python main.py

# Web app
uv run uvicorn server:app --reload           # backend (:8000)
cd frontend && npm install && npm run dev    # frontend (:5173)
```

The backend reads no key from the environment — each request carries the user's OpenRouter key in
the `X-OpenRouter-Key` header. Its **Implementation** panel shows which of the eight functions are
live and which are still missing.

### Where your documents are stored

Uploads go into a local on-disk **Qdrant** store at `project/qdrant_data` (gitignored), which
persists across restarts. Override it with `RAG_DB_PATH`, and the collection name with
`RAG_COLLECTION` (default `documents`). To wipe it, use `DELETE /documents` in the app, or delete
the folder while the server is stopped. `main.py` and the notebook use an ephemeral in-memory store
instead.

Point the frontend at a different backend with `VITE_API_BASE`.

See `CLAUDE.md` for the full architecture.
