# WE6_ProjectRAG

A teaching-oriented Retrieval-Augmented Generation (RAG) system for a **Compliance Q&A** project:
ask questions about company policy and get answers grounded in a provided document set.

## The course notebook

The project is delivered as a Jupyter notebook in which students implement the core of the
pipeline themselves:

- **`notebook/compliance_rag_project.ipynb`** — the student notebook (exercise stubs + tests).
  Designed to run in **Google Colab**; works in local Jupyter too.

  Credentials come from **Colab Secrets** (🔑 panel, "Notebook access" ON) — never pasted into a
  cell:
  - `GITHUB_TOKEN` — a GitHub Personal Access Token to clone the (private) course repo in Colab.
    The setup cell clones a specific branch (`REPO_BRANCH`, default `notebook`) — change it to
    `main` once the project is merged there.
  - `OPENROUTER_API_KEY` — *optional*, only for the live demo at the end. Locally you can instead
    `export OPENROUTER_API_KEY=...` before launching Jupyter.
- **`notebook/compliance_rag_project.solution.ipynb`** — the fully-worked answer key.
- **`notebook/build_notebook.py`** — regenerates both:
  ```bash
  uv run python notebook/build_notebook.py              # student notebook
  uv run python notebook/build_notebook.py --solution   # answer key
  ```

Students implement each function in the notebook and *monkey-patch* it onto the real project code,
so the provided scaffolding and tests run against their implementation. At the end of each part an
export cell writes their functions to `solutions/part1_ingestion.py` and
`solutions/part2_retrieval.py`.

## Running the full app on your own implementations

Drop your two exported files into `solutions/`. On startup `main.py` / `server.py` call
`solutions.apply()`, which monkey-patches your functions onto the live app (a no-op on a fresh
checkout, so the reference implementation always keeps the app working).

```bash
uv sync
uv run uvicorn server:app --reload          # backend (:8000)
cd frontend && npm install && npm run dev    # frontend (:5173)
```

See `CLAUDE.md` for the full architecture and commands.
