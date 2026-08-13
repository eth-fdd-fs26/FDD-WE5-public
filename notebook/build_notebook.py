"""Generate the course notebook for the Compliance Q&A RAG project.

The notebook is built programmatically so it stays diffable and regenerable. Each exercise
carries both a *stub* (what students receive) and a *solution* (the repo's reference body):

    uv run python notebook/build_notebook.py                  # student notebook (stubs)
    uv run python notebook/build_notebook.py --solution       # fully-runnable answer key
    uv run python notebook/build_notebook.py -o some/path.ipynb

The two flavours share every markdown/test/patch cell; only the exercise cells differ. The
solution flavour is what we execute offline to verify the whole notebook runs green.
"""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

SOLUTION = "--solution" in sys.argv

_out_flag = "-o" in sys.argv
if _out_flag:
    OUTPUT = Path(sys.argv[sys.argv.index("-o") + 1])
else:
    name = "compliance_rag_project.solution.ipynb" if SOLUTION else "compliance_rag_project.ipynb"
    OUTPUT = Path(__file__).resolve().parent / name

cells: list = []


def md(src: str) -> None:
    cells.append(new_markdown_cell(src.strip("\n")))


def code(src: str) -> None:
    cells.append(new_code_cell(src.strip("\n")))


def exercise(stub: str, solution: str) -> None:
    cells.append(new_code_cell((solution if SOLUTION else stub).strip("\n")))


# --------------------------------------------------------------------------- #
# Inline SVG diagram builders (rendered inside markdown cells). A shared palette
# keeps the diagrams and the HTML output helpers visually consistent.
# --------------------------------------------------------------------------- #
INK, MUTE, TEAL, AMBER, LINE, BG, SLATE = (
    "#1f2933", "#647084", "#0f766e", "#b45309", "#e3e8ef", "#f8fafc", "#334155",
)


def _t(x, y, s, size=11, fill=MUTE, weight="400", anchor="middle"):
    return (f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{size}" '
            f'font-weight="{weight}" fill="{fill}">{s}</text>')


def _box(x, w, label, sub="", y=46, h=54, stroke=TEAL, fill="#ffffff"):
    cx = x + w / 2
    out = (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{fill}" '
           f'stroke="{stroke}" stroke-width="1.6"/>')
    if sub:
        out += _t(cx, y + 23, label, 13.5, INK, "700")
        out += _t(cx, y + 40, sub, 10.5, MUTE)
    else:
        out += _t(cx, y + h / 2 + 5, label, 13.5, INK, "700")
    return out


def _arrow(x1, y1, x2, y2, dashed=False):
    dash = ' stroke-dasharray="5 4"' if dashed else ""
    return (f'<path d="M{x1},{y1} L{x2},{y2}" stroke="{MUTE}" stroke-width="1.6" '
            f'fill="none" marker-end="url(#ah)"{dash}/>')


def _elbow(points, dashed=False):
    d = " ".join((f"M{x},{y}" if i == 0 else f"L{x},{y}") for i, (x, y) in enumerate(points))
    dash = ' stroke-dasharray="5 4"' if dashed else ""
    return (f'<path d="{d}" stroke="{MUTE}" stroke-width="1.6" fill="none" '
            f'marker-end="url(#ah)"{dash}/>')


def _svg(inner, w, h):
    return (f'<div style="text-align:center;margin:10px 0">'
            f'<svg viewBox="0 0 {w} {h}" width="100%" style="max-width:{w}px;'
            f'font-family:ui-sans-serif,system-ui,sans-serif">'
            f'<defs><marker id="ah" markerWidth="9" markerHeight="9" refX="7" refY="3" '
            f'orient="auto"><path d="M0,0 L8,3 L0,6 Z" fill="{MUTE}"/></marker></defs>'
            f'{inner}</svg></div>')


DIAGRAM_ARCH = _svg(
    _t(20, 22, "1 · WRITE — ingestion", 12, SLATE, "700", "start")
    + _box(20, 150, "Documents", "policies", stroke=SLATE)
    + _box(220, 150, "Chunk", "split into passages")
    + _box(410, 150, "Embed", "text → vectors")
    + _box(600, 150, "Vector store", "Qdrant", stroke=AMBER)
    + _arrow(170, 73, 220, 73) + _arrow(370, 73, 410, 73) + _arrow(560, 73, 600, 73)
    + _t(20, 216, "2 · READ — query", 12, SLATE, "700", "start")
    + _box(20, 150, "Question", y=240, stroke=SLATE)
    + _box(220, 150, "Hybrid search", "keyword + embedding", y=240)
    + _box(470, 120, "LLM", y=240, stroke=SLATE)
    + _box(640, 120, "Answer", "+ citations", y=240, stroke=AMBER)
    + _arrow(170, 267, 220, 267) + _arrow(370, 267, 470, 267) + _arrow(590, 267, 640, 267)
    + _elbow([(675, 100), (675, 150), (295, 150), (295, 240)], dashed=True)
    + _t(485, 143, "stored chunks loaded into the retriever", 10, MUTE, "600"),
    820, 300,
)

DIAGRAM_INGEST = _svg(
    _box(20, 160, "load", "file → Document", stroke=SLATE)
    + _box(210, 160, "CHUNK", "Document → texts")
    + _box(400, 160, "EMBED", "texts → vectors")
    + _box(590, 160, "store", "→ vector DB", stroke=AMBER)
    + _arrow(180, 73, 210, 73) + _arrow(370, 73, 400, 73) + _arrow(560, 73, 590, 73)
    + _t(100, 118, "loaders.py", 10, MUTE, "600")
    + _t(290, 118, "chunking.py  ✏️ you", 10, TEAL, "700")
    + _t(480, 118, "ingest_document  ✏️ you", 10, TEAL, "700")
    + _t(670, 118, "db_manager.py", 10, MUTE, "600"),
    770, 140,
)


def _wb(i, x):
    return (f'<rect x="{x}" y="58" width="54" height="32" rx="6" fill="{BG}" stroke="{LINE}"/>'
            + _t(x + 27, 79, f"w{i}", 12, INK, "600"))


def _chunkbar(x, w, y, label, color):
    return (f'<rect x="{x}" y="{y}" width="{w}" height="24" rx="6" fill="{color}" '
            f'fill-opacity="0.16" stroke="{color}" stroke-width="1.4"/>'
            + _t(x + w / 2, y + 16, label, 10.5, color, "700"))


DIAGRAM_SLIDING = _svg(
    _t(20, 28, "step = chunk_size − overlap = 3   (chunk_size=4, overlap=1)", 11, SLATE, "600", "start")
    + "".join(_wb(i, 20 + i * 62) for i in range(9))
    + _chunkbar(16, 250, 104, "chunk 0  ·  w0–w3", TEAL)
    + _chunkbar(202, 250, 134, "chunk 1  ·  w3–w6", AMBER)
    + _chunkbar(388, 188, 164, "chunk 2  ·  w6–w8", SLATE),
    600, 200,
)

DIAGRAM_RETRIEVAL = _svg(
    _box(16, 124, "Question", y=82, stroke=SLATE)
    + _box(190, 168, "Keyword search", "BM25  ✏️", y=18)
    + _box(190, 168, "Embedding search", "cosine  ✏️", y=138)
    + _box(400, 116, "Fuse", "RRF  ✏️", y=82, stroke=AMBER)
    + _box(548, 116, "Rerank", "bonus", y=82, stroke=LINE)
    + _box(696, 92, "LLM", y=82, stroke=SLATE)
    + _box(800, 92, "Answer", y=82, stroke=AMBER)
    + _arrow(140, 100, 190, 58) + _arrow(140, 114, 190, 178)
    + _arrow(358, 58, 400, 100) + _arrow(358, 178, 400, 120)
    + _arrow(516, 109, 548, 109) + _arrow(664, 109, 696, 109) + _arrow(788, 109, 800, 109),
    900, 214,
)

DIAGRAM_COSINE = _svg(
    f'<line x1="60" y1="170" x2="250" y2="50" stroke="{TEAL}" stroke-width="2.4" marker-end="url(#ah)"/>'
    + f'<line x1="60" y1="170" x2="270" y2="120" stroke="{AMBER}" stroke-width="2.4" marker-end="url(#ah)"/>'
    + f'<path d="M120,135 A 62 62 0 0 1 128 116" fill="none" stroke="{MUTE}" stroke-width="1.4"/>'
    + _t(150, 132, "θ", 14, INK, "700")
    + _t(255, 42, "query vector q", 12, TEAL, "700", "start")
    + _t(276, 122, "chunk vector d", 12, AMBER, "700", "start")
    + _t(60, 190, "cos(q, d) = (q · d) / (‖q‖ ‖d‖)  —  small angle → score near 1", 11, SLATE, "600", "start"),
    520, 210,
)


def _rrf_list(title, items, accent):
    rows = "".join(
        f'<div style="display:flex;gap:8px;padding:3px 0">'
        f'<span style="color:{MUTE};font-family:ui-monospace,monospace;font-size:11px">#{i}</span>'
        f'<span style="color:{INK};font-weight:600;font-size:13px">{x}</span></div>'
        for i, x in enumerate(items, 1))
    return (f'<div style="border:1px solid {LINE};border-top:3px solid {accent};border-radius:8px;'
            f'padding:8px 12px;min-width:130px"><div style="font-size:11px;color:{accent};'
            f'font-weight:700;text-transform:uppercase;letter-spacing:.04em;margin-bottom:2px">{title}</div>{rows}</div>')


DIAGRAM_RRF = (
    '<div style="display:flex;gap:14px;align-items:center;justify-content:center;flex-wrap:wrap;'
    'margin:12px 0;font-family:ui-sans-serif,system-ui,sans-serif">'
    + _rrf_list("keyword", ["chunk A", "chunk B"], TEAL)
    + _rrf_list("embedding", ["chunk B", "chunk C"], TEAL)
    + f'<div style="font-size:24px;color:{MUTE}">→</div>'
    + _rrf_list("fused (RRF)", ["chunk B · 1/60 + 1/61", "chunk A · 1/60", "chunk C · 1/61"], AMBER)
    + '</div>')


# =========================================================================== #
# PART 0 — Setup & orientation
# =========================================================================== #
md(r'''
# Compliance Q&A — a Retrieval-Augmented Generation project

Welcome! In this project you will build a **RAG application** that lets employees ask questions
about company policy — *"How many days of annual leave do I get?"*, *"Can I accept a CHF 120 gift
from a vendor?"* — and get answers grounded in a real document set.

You will implement the **core of the pipeline yourself**. The surrounding scaffolding (data
models, the vector store, the AI clients, a web app) is provided, so you can focus on the ideas
that make RAG work.

## How this notebook works

The project ships as a normal Python repo. Rather than editing those files directly, you will
write each function **here, in the notebook**, and *monkey-patch* it onto the real project code:

```python
import chunking
def sliding_window(...): ...
chunking.sliding_window = sliding_window   # <- your version now runs everywhere
```

Because the rest of the code looks your function up by name at call-time, your implementation
immediately flows through the whole pipeline and the provided tests. At the end of each part an
**export cell** writes your functions to a file you can drop into the repo to run the full app.

## Roadmap

- **Part 1 — Ingestion:** turn documents into searchable, embedded chunks (chunking, embeddings,
  metadata).
- **Part 2 — RAG core:** answer a question (keyword search, embedding search, fusion, reranking,
  and finally calling the LLM with the retrieved context).

Each step is: a short explanation → a function for **you** to implement → a **test cell** that
tells you whether it works.

### The whole system at a glance
''' + DIAGRAM_ARCH + r'''

The **write** path (top) is built in **Part 1**; the **read** path (bottom) in **Part 2**. The two
meet at the **vector store**: ingestion writes embedded chunks into it, and every query loads those
chunks back to search them.
''')

md(r'''
## 0.1 — Setup

This notebook reads its credentials from **Colab Secrets** — never paste keys into a cell. Open the
**🔑 key icon** in the left sidebar ("Secrets") and add the following, toggling **"Notebook access"
ON** for each:

| Secret name | What it's for | Required? |
|---|---|---|
| `GITHUB_TOKEN` | clone the (private) course repo | only in Colab, if the repo is private |
| `OPENROUTER_API_KEY` | the *optional* live demo at the end | optional |

> A `GITHUB_TOKEN` is a GitHub **Personal Access Token** (github.com → Settings → Developer
> settings → Tokens) with read access to the course repo. Secrets are private to your account and
> are **not** saved inside the notebook file.

Now run the cell below. In **Colab** it clones the project (the `REPO_BRANCH` branch), installs
dependencies, and puts the code on the import path. Running **locally inside the repo**, it just
makes the repo importable.
''')

# The solution notebook stays inside the private repo; the student notebook is what gets
# published to the public repo, so it must clone from there instead.
_setup_cell = r'''
import os
import sys
import subprocess

IN_COLAB = "google.colab" in sys.modules

# If the course gives you a different (e.g. public) template URL, change this.
REPO_URL = "https://github.com/eth-fdd-fs26/FDD-WE5-private.git"
# The branch that holds the project files (this notebook + the solutions/ loader).
# Change to "main" once the project has been merged there.
REPO_BRANCH = "project"
REPO_DIR = "FDD-WE5-private"


def colab_secret(name):
    """Read a Colab secret by name; return None if missing or access not granted."""
    try:
        from google.colab import userdata
        return userdata.get(name)
    except Exception:
        return None


if IN_COLAB:
    # Authenticate the clone with the GITHUB_TOKEN Colab secret (see the table above).
    token = colab_secret("GITHUB_TOKEN")
    if not token:
        print("⚠️  No GITHUB_TOKEN secret found. If the repo is private the clone will fail —\n"
              "    add it via the 🔑 Secrets panel and toggle 'Notebook access' on, then re-run.")
    # Use the x-access-token form (the documented way to clone with a PAT).
    url = REPO_URL.replace("https://", f"https://x-access-token:{token}@") if token else REPO_URL
    if not os.path.isdir(REPO_DIR):
        # --branch pins the correct branch; --single-branch keeps the clone small.
        # We capture output (rather than check=True) so a failure never echoes the
        # token back in a traceback — we redact it before showing anything.
        proc = subprocess.run(
            ["git", "clone", "--branch", REPO_BRANCH, "--single-branch", url, REPO_DIR],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or "").strip()
            if token:
                detail = detail.replace(token, "***")  # never print the secret
            raise RuntimeError(
                "git clone failed (exit %d):\n%s\n\nCommon causes:\n"
                "  - the GITHUB_TOKEN lacks access: a classic PAT needs the 'repo' scope "
                "(fine-grained: Contents = Read on this repo);\n"
                "  - the token expired, or isn't authorised for the eth-fdd-fs26 org;\n"
                "  - REPO_BRANCH (%r) doesn't exist on the remote." % (proc.returncode, detail, REPO_BRANCH)
            )
    %pip install -q qdrant-client rank-bm25 numpy openai pypdf rich
    sys.path.insert(0, os.path.abspath(REPO_DIR))
    os.chdir(REPO_DIR)
else:
    # Local: find the repo root (the folder that contains chunking.py), make it
    # importable, and work from there — so it doesn't matter whether you launched
    # the notebook from the repo root or from the notebook/ subfolder.
    root = os.path.abspath(".")
    while root != os.path.dirname(root) and not os.path.exists(os.path.join(root, "chunking.py")):
        root = os.path.dirname(root)
    sys.path.insert(0, root)
    os.chdir(root)

print("Setup complete. Running in Colab:", IN_COLAB)
print("Working directory:", os.getcwd())
'''
if not SOLUTION:
    _setup_cell = _setup_cell.replace(
        'REPO_URL = "https://github.com/eth-fdd-fs26/FDD-WE5-private.git"',
        'REPO_URL = "https://github.com/eth-fdd-fs26/FDD-WE5-public.git"',
    ).replace(
        'REPO_DIR = "FDD-WE5-private"',
        'REPO_DIR = "FDD-WE5-public"',
    )
code(_setup_cell)

md(r'''
## 0.2 — An (optional) OpenRouter key

Everything in this notebook runs **fully offline** using small mock AI clients, so you do not need
an API key to do the project or pass the tests.

If you *do* want to see the real system answer a question at the end, add your
[OpenRouter](https://openrouter.ai) key as the Colab secret **`OPENROUTER_API_KEY`** (🔑 Secrets
panel, "Notebook access" ON). The cell below loads it from your secrets — locally, you can instead
`export OPENROUTER_API_KEY=...` before launching Jupyter. The key is read straight into the
environment and never written into the notebook.
''')

code(r'''
# Pull the OpenRouter key from Colab Secrets (no-op locally / if the secret is absent).
_k = colab_secret("OPENROUTER_API_KEY") if IN_COLAB else None
if _k:
    os.environ["OPENROUTER_API_KEY"] = _k

HAS_KEY = bool(os.environ.get("OPENROUTER_API_KEY"))
print("Live API key available:", HAS_KEY)
print("(Everything below works offline with mocks regardless.)")
''')

md(r'''
## 0.3 — Imports

We import the provided scaffolding once. Notice we reuse the project's own test harness
(`TestSuite`) and offline doubles (`MockEmbedder`, `MockLLM`) — the exact same tools the course
maintainers use.
''')

code(r'''
import inspect
import pathlib

import numpy as np
from rank_bm25 import BM25Okapi

# Provided scaffolding from the repo:
import chunking
import loaders
from ingestion_core import IngestionCore
from db_manager import DBManager
from rag_core import RAGCore
from models import SearchType
from retrieval_core import RetrievalCore, _tokenize
from models import Chunk, Document

# Offline test doubles + the friendly test runner (also from the repo):
from tests.mocks import MockEmbedder, MockLLM
from tests.harness import TestSuite

print("Imports OK — ready to build the pipeline.")
''')

md(r'''
## 0.4 — Monkey-patching in 60 seconds

Before relying on it, see the trick work. We replace a function and watch the change ripple
through code that *calls* it.
''')

code(r'''
def _demo(): return "original"
print("before:", _demo())

# Rebind the name — any later call sees the new version.
def _demo(): return "patched!"
print("after: ", _demo())
''')

md(r'''
## 0.5 — Pretty output helpers

To make results easy to read, we render them as small HTML cards instead of plain text. Just run
this cell — you'll call these helpers (`show_documents`, `show_chunks`, `show_results`,
`show_answer`) throughout the notebook.
''')

code(r'''
from IPython.display import HTML, display

# A small, consistent visual language for outputs in this notebook.
_INK, _MUTE, _TEAL, _AMBER, _LINE, _BG = "#1f2933", "#647084", "#0f766e", "#b45309", "#e3e8ef", "#f8fafc"
_FONT = "ui-sans-serif,system-ui,sans-serif"
_MONO = "ui-monospace,SFMono-Regular,Menlo,monospace"

def _card(inner, accent=_TEAL):
    return (f'<div style="border:1px solid {_LINE};border-left:4px solid {accent};border-radius:10px;'
            f'padding:14px 16px;margin:8px 0;background:#fff;font-family:{_FONT}">{inner}</div>')

def _clip(text, n):
    return (text[:n] + "…") if len(text) > n else text

def show_documents(docs):
    head = (f'<tr style="color:{_MUTE};font-size:11px;text-transform:uppercase;letter-spacing:.05em">'
            f'<th style="text-align:left;padding:6px 10px">Document</th>'
            f'<th style="text-align:left;padding:6px 10px">Source</th>'
            f'<th style="text-align:right;padding:6px 10px">Words</th></tr>')
    rows = "".join(
        f'<tr><td style="padding:6px 10px;border-top:1px solid {_LINE};font-weight:600;color:{_INK}">{d.metadata["document_title"]}</td>'
        f'<td style="padding:6px 10px;border-top:1px solid {_LINE};color:{_MUTE};font-family:{_MONO};font-size:12px">{d.metadata["source"]}</td>'
        f'<td style="padding:6px 10px;border-top:1px solid {_LINE};text-align:right;color:{_INK}">{len(d.text.split())}</td></tr>'
        for d in docs)
    title = f'<div style="font-weight:700;color:{_INK};margin-bottom:6px">📚 {len(docs)} policy documents</div>'
    display(HTML(_card(title + f'<table style="border-collapse:collapse;width:100%">{head}{rows}</table>')))

def show_chunks(chunks, title="Chunks", limit=4):
    items = "".join(
        f'<div style="background:{_BG};border:1px solid {_LINE};border-radius:8px;padding:8px 10px;margin:6px 0">'
        f'<span style="font-family:{_MONO};font-size:11px;color:{_TEAL};font-weight:700">#{c.index}</span> '
        f'<span style="color:{_INK};font-size:13px">{_clip(c.text, 150)}</span></div>'
        for c in chunks[:limit])
    more = f'<div style="color:{_MUTE};font-size:12px">… and {len(chunks)-limit} more</div>' if len(chunks) > limit else ""
    display(HTML(_card(f'<div style="font-weight:700;color:{_INK};margin-bottom:4px">🧩 {title} · {len(chunks)} total</div>{items}{more}')))

def show_chunking(whole, windows):
    def col(name, chunks, accent):
        items = "".join(
            f'<div style="background:{_BG};border:1px solid {_LINE};border-radius:8px;padding:8px;margin:6px 0;font-size:12px;color:{_INK}">{_clip(ch, 120)}</div>'
            for ch in chunks[:4])
        more = f'<div style="color:{_MUTE};font-size:12px">… {len(chunks)-4} more</div>' if len(chunks) > 4 else ""
        return f'<div style="flex:1"><div style="font-weight:700;color:{accent};margin-bottom:4px">{name} · {len(chunks)} chunk(s)</div>{items}{more}</div>'
    display(HTML(_card(f'<div style="display:flex;gap:16px">{col("whole_document", whole, _AMBER)}{col("sliding_window", windows, _TEAL)}</div>')))

def show_results(results, title="Results"):
    if not results:
        display(HTML(_card("<em>No results.</em>", _AMBER)))
        return
    mx = max(s for _, s in results) or 1.0
    rows = []
    for rank, (c, s) in enumerate(results, 1):
        pct = max(4.0, 100 * s / mx)
        rows.append(
            f'<div style="display:flex;gap:10px;align-items:flex-start;padding:8px 0;border-top:1px solid {_LINE}">'
            f'<div style="color:{_MUTE};font-weight:700;width:20px">{rank}</div><div style="flex:1">'
            f'<div style="font-size:11px;color:{_MUTE};font-family:{_MONO}">{c.metadata.get("source", "?")}</div>'
            f'<div style="color:{_INK};font-size:13px;margin:2px 0 4px">{_clip(c.text, 150)}</div>'
            f'<div style="background:{_BG};border-radius:6px;height:6px;overflow:hidden">'
            f'<div style="width:{pct:.0f}%;height:6px;background:{_TEAL}"></div></div></div>'
            f'<div style="font-family:{_MONO};color:{_TEAL};font-weight:700">{s:.3f}</div></div>')
    display(HTML(_card(f'<div style="font-weight:700;color:{_INK};margin-bottom:2px">🔎 {title}</div>' + "".join(rows))))

def show_answer(answer, sources):
    src = "".join(
        f'<li style="margin:5px 0;color:{_MUTE};font-size:13px"><code style="color:{_TEAL}">{c.metadata.get("source", "?")}</code> '
        f'<span style="color:{_TEAL};font-weight:700">{s:.3f}</span> — {_clip(c.text, 90)}</li>'
        for c, s in sources)
    body = (f'<div style="font-size:11px;color:{_MUTE};text-transform:uppercase;letter-spacing:.05em">Answer</div>'
            f'<div style="color:{_INK};font-size:15px;line-height:1.55;margin:6px 0 10px">{answer}</div>'
            f'<details><summary style="cursor:pointer;color:{_TEAL};font-weight:700">{len(sources)} sources</summary>'
            f'<ul style="margin:8px 0 0;padding-left:18px">{src}</ul></details>')
    display(HTML(_card(body, _AMBER)))

print("Display helpers ready: show_documents, show_chunks, show_chunking, show_results, show_answer")
''')

# =========================================================================== #
# PART 1 — INGESTION
# =========================================================================== #
md(r'''
---
# Part 1 — Document ingestion pipeline

Ingestion is the **write side** of RAG. It turns raw documents into searchable chunks:

```
load  ->  CHUNK  ->  EMBED  ->  store
```

1. **load** — read a file into a `Document` (provided), attaching metadata.
2. **chunk** — split the text into focused passages (**you implement this**).
3. **embed** — turn each chunk into a vector (**you wire this up**).
4. **store** — persist the embedded chunks into the vector database (provided).
''' + DIAGRAM_INGEST + r'''

The ✏️ marked steps are the ones **you** implement. Let's look at the documents we'll work with.
''')

code(r'''
docs = loaders.load_directory("data")
show_documents(docs)
''')

# --- 1.1 Chunking ---------------------------------------------------------- #
md(r'''
## 1.1 — Chunking

An embedding model turns a piece of text into **one** vector. Embed a whole 800-word policy as a
single vector and it becomes a blurry average of every idea in it — a question about one clause has
to compete with the noise of the entire document.

The fix is **chunking**: split the document into smaller passages so each idea gets its own vector.
We use a **sliding window** over words, with a little **overlap** so a sentence straddling a
boundary isn't lost.

''' + DIAGRAM_SLIDING + r'''

Notice how `w3` and `w6` each sit in **two** chunks — that's the overlap, and it's what keeps a
sentence spanning a boundary from being lost.

> **Reflection:** what goes wrong if `overlap` is large relative to `chunk_size`? What if it's 0?

### ✏️ Your turn: implement `sliding_window`
''')

exercise(
    stub=r'''
def sliding_window(text, chunk_size=200, overlap=40):
    """Split `text` into overlapping windows of words.

    Steps:
      1. Validate inputs: chunk_size > 0, overlap >= 0, overlap < chunk_size (raise ValueError).
      2. words = text.split(); return [] if there are no words.
      3. step = chunk_size - overlap. Slide `start` over range(0, len(words), step);
         take words[start:start + chunk_size] and join them with a single space.
      4. Stop once a window reaches the end, so you don't emit a duplicate trailing chunk.

    Returns: list[str] of chunk texts.
    """
    # ✏️ YOUR CODE HERE
    raise NotImplementedError
''',
    solution=r'''
def sliding_window(text, chunk_size=200, overlap=40):
    """Split `text` into overlapping windows of words."""
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if overlap < 0:
        raise ValueError(f"overlap must be >= 0, got {overlap}")
    if overlap >= chunk_size:
        raise ValueError(f"overlap ({overlap}) must be smaller than chunk_size ({chunk_size})")

    words = text.split()
    if not words:
        return []

    step = chunk_size - overlap
    chunks = []
    for start in range(0, len(words), step):
        window = words[start:start + chunk_size]
        if not window:
            break
        chunks.append(" ".join(window))
        if start + chunk_size >= len(words):
            break
    return chunks
''',
)

md(r'''
Now **monkey-patch** your function onto the `chunking` module. The whole pipeline calls
`chunking.sliding_window`, so this one line makes your version the one that runs everywhere.
''')

code(r'''
chunking.sliding_window = sliding_window
print("Patched chunking.sliding_window ->", chunking.sliding_window.__name__)
''')

code(r'''
suite = TestSuite("Part 1.1 — sliding_window")

@suite.case("sliding_window", "overlap shares words across consecutive windows")
def _():
    assert sliding_window("a b c d e f", chunk_size=4, overlap=1) == ["a b c d", "d e f"]

@suite.case("sliding_window", "text shorter than the window -> a single chunk")
def _():
    assert sliding_window("a b", chunk_size=4, overlap=1) == ["a b"]

@suite.case("sliding_window", "empty text -> no chunks")
def _():
    assert sliding_window("   ", chunk_size=4, overlap=1) == []

@suite.case("sliding_window", "overlap >= chunk_size is rejected")
def _():
    try:
        sliding_window("a b c", chunk_size=2, overlap=2)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass

suite.run()
''')

md(r'''
**See why chunking matters.** Compare the "no chunking" baseline against your sliding window on a
real policy — notice how many focused passages you get instead of one giant blob.
''')

code(r'''
sample = next(d for d in docs if "leave" in d.metadata["source"])
whole = chunking.chunk_text(sample.text, strategy="whole_document")
windows = chunking.chunk_text(sample.text, chunk_size=80, overlap=15, strategy="sliding_window")
show_chunking(whole, windows)
''')

# --- 1.2 Embeddings -------------------------------------------------------- #
md(r'''
## 1.2 — Embeddings & the simple vector database

An **embedding** maps text to a vector so that *similar meanings land near each other*. Once every
chunk is a vector, "search" becomes "find the nearest vectors to the query vector".

You'll wire up the full ingestion step on `IngestionCore`: chunk the document, wrap each passage in
a `Chunk`, embed them all in one batched call, and store them. The `embedder`, `db`, and chunking
config are already on `self`.

- `chunking.chunk_text(text, chunk_size=..., overlap=..., strategy=...) -> list[str]`
- `Chunk(document_id=..., index=..., text=..., metadata=...)`
- `self.embedder.embed_batch(list_of_texts) -> list[list[float]]` (order preserved)
- `self.db.ensure_collection(name, vector_size=...)` then `self.db.insert_document(name, document, chunks)`

### ✏️ Your turn: implement `ingest_document`
''')

exercise(
    stub=r'''
def ingest_document(self, document):
    """Run load->CHUNK->EMBED->store for one Document. Return the list of stored Chunks.

    Steps:
      1. texts = chunking.chunk_text(document.text, chunk_size=self.chunk_size,
                                     overlap=self.overlap, strategy=self.strategy)
         If texts is empty, return [].
      2. Build one Chunk per text:
             Chunk(document_id=document.id, index=i, text=text,
                   metadata=dict(document.metadata))
      3. embeddings = self.embedder.embed_batch([c.text for c in chunks])
         Assign each embedding onto its chunk (chunk.embedding = ...).
      4. self.db.ensure_collection(self.collection_name, vector_size=len(embeddings[0]))
         self.db.insert_document(self.collection_name, document, chunks)
      5. return chunks
    """
    # ✏️ YOUR CODE HERE
    raise NotImplementedError
''',
    solution=r'''
def ingest_document(self, document):
    """Run load->CHUNK->EMBED->store for one Document. Return the list of stored Chunks."""
    texts = chunking.chunk_text(
        document.text,
        chunk_size=self.chunk_size,
        overlap=self.overlap,
        strategy=self.strategy,
    )
    if not texts:
        return []

    chunks = [
        Chunk(document_id=document.id, index=i, text=text, metadata=dict(document.metadata))
        for i, text in enumerate(texts)
    ]

    embeddings = self.embedder.embed_batch([chunk.text for chunk in chunks])
    for chunk, embedding in zip(chunks, embeddings):
        chunk.embedding = embedding

    self.db.ensure_collection(self.collection_name, vector_size=len(embeddings[0]))
    self.db.insert_document(self.collection_name, document, chunks)
    return chunks
''',
)

code(r'''
IngestionCore.ingest_document = ingest_document
print("Patched IngestionCore.ingest_document")
''')

code(r'''
suite = TestSuite("Part 1.2 — ingest_document")

def _fresh_core():
    db = DBManager()  # in-memory, ephemeral
    core = IngestionCore(MockEmbedder([1.0, 0.0, 0.0]), db, collection_name="t",
                         chunk_size=5, overlap=1)
    return db, core

_doc = Document(text="one two three four five six seven eight nine ten",
                metadata={"source": "x.md", "document_title": "X"})

@suite.case("ingest_document", "every chunk comes back embedded")
def _():
    _db, core = _fresh_core()
    chunks = core.ingest_document(_doc)
    assert chunks and all(c.embedding is not None for c in chunks)

@suite.case("ingest_document", "chunk indices are sequential from 0")
def _():
    _db, core = _fresh_core()
    chunks = core.ingest_document(_doc)
    assert [c.index for c in chunks] == list(range(len(chunks)))

@suite.case("ingest_document", "chunks are persisted and re-readable from the store")
def _():
    db, core = _fresh_core()
    chunks = core.ingest_document(_doc)
    assert len(db.get_all_chunks("t")) == len(chunks)

@suite.case("ingest_document", "document metadata rides onto every chunk")
def _():
    _db, core = _fresh_core()
    chunks = core.ingest_document(_doc)
    assert all(c.metadata.get("source") == "x.md" for c in chunks)

suite.run()
''')

# --- 1.3 Metadata ---------------------------------------------------------- #
md(r'''
## 1.3 — Metadata

When we load a file we attach **metadata**: where it came from (`source`) and a human title
(`document_title`). That metadata travels with every chunk, and later lets retrieval *filter* —
"only search the leave policy" — so unrelated documents can't crowd out the right answer.

You'll implement the title-deriving helper: use the first Markdown `# heading` if there is one,
otherwise fall back to a tidied-up file stem (`leave_policy` → `"Leave Policy"`).

### ✏️ Your turn: implement `derive_title`
''')

exercise(
    stub=r'''
def derive_title(text, stem):
    """Return a document title.

    Steps:
      1. Scan the lines of `text`; if a line (stripped) starts with "# ", return the rest of it.
      2. Otherwise fall back to the file stem: replace "_" and "-" with spaces and Title-Case it.
    """
    # ✏️ YOUR CODE HERE
    raise NotImplementedError
''',
    solution=r'''
def derive_title(text, stem):
    """Return a document title: first Markdown heading, else a tidied file stem."""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()
    return stem.replace("_", " ").replace("-", " ").title()
''',
)

code(r'''
loaders._derive_title = derive_title
print("Patched loaders._derive_title")
''')

code(r'''
suite = TestSuite("Part 1.3 — derive_title")

@suite.case("derive_title", "uses the first Markdown heading when present")
def _():
    assert derive_title("# Leave Policy\n\nText...", "leave_policy") == "Leave Policy"

@suite.case("derive_title", "falls back to a title-cased stem")
def _():
    assert derive_title("no heading here", "leave_and_absence") == "Leave And Absence"

suite.run()
''')

md(r'''
**Metadata makes search precise.** Here is the payoff: given a pile of chunks from *all* policies,
a metadata filter narrows the field to one document *before* any scoring happens. (The filter
helper is provided — you'll use it for real in Part 2.)
''')

code(r'''
all_chunks = []
_db = DBManager()
_core = IngestionCore(MockEmbedder([1.0, 0.0, 0.0]), _db, collection_name="demo",
                      chunk_size=120, overlap=20)
for d in docs:
    all_chunks.extend(_core.ingest_document(d))

only_leave = RetrievalCore._apply_metadata_filter(all_chunks, {"source": "leave_and_absence_policy.md"})
display(HTML(_card(
    f'<b style="color:{_INK}">Metadata filter</b> '
    f'<code style="color:{_TEAL}">{{"source": "leave_and_absence_policy.md"}}</code><br>'
    f'corpus: <b>{len(all_chunks)}</b> chunks &nbsp;→&nbsp; '
    f'after filter: <b style="color:{_TEAL}">{len(only_leave)}</b> chunks')))
show_chunks(only_leave, title="Leave-policy chunks (after filtering)")
''')

# --- 1.4 Export ------------------------------------------------------------ #
md(r'''
## 1.4 — Export your Part 1 functions

This writes your three functions to `solutions/part1_ingestion.py`. Dropped into the repo, the app
will run on *your* ingestion code (see Part 3). **Run your implementation cells above first**, then
run this.
''')

code(r'''
HEADER_P1 = (
    '"""Part 1 — exported from the notebook. Do not edit by hand; re-export instead."""\n'
    'from __future__ import annotations\n'
    'import chunking\n'
    'from models import Chunk\n\n\n'
)

_funcs_p1 = [sliding_window, ingest_document, derive_title]
_body = HEADER_P1 + "\n\n".join(inspect.getsource(f) for f in _funcs_p1)

_path = pathlib.Path("solutions/part1_ingestion.py")
_path.parent.mkdir(exist_ok=True)
_path.write_text(_body)
print(f"Wrote {_path} ({len(_funcs_p1)} functions).")

try:
    from google.colab import files
    files.download(str(_path))
except Exception:
    pass
''')

md(r'''
### Bonus (optional): smarter vector stores

The provided store keeps chunks in a flat list and compares the query against *every* vector.
That's fine for a few thousand chunks but doesn't scale. Real systems use **approximate
nearest-neighbour** indexes such as **HNSW** (Hierarchical Navigable Small World graphs), which
trade a tiny bit of accuracy for enormous speed. Qdrant (the store under `DBManager`) uses HNSW
internally. *Bonus:* read how HNSW builds a layered graph and why it's so much faster than a linear
scan.
''')

# =========================================================================== #
# PART 2 — RAG CORE
# =========================================================================== #
md(r'''
---
# Part 2 — The RAG core

Now the **read side**: take a question and produce a grounded answer.

''' + DIAGRAM_RETRIEVAL + r'''

You'll build each retrieval strategy (the ✏️ steps), combine them, optionally rerank, and finally
hand the chosen chunks to the LLM.
''')

# --- 2.1 Embedding search -------------------------------------------------- #
md(r'''
## 2.1 — Embedding (semantic) search

Semantic search ranks chunks by how close their vectors are to the **query's** vector. The standard
closeness measure is **cosine similarity** — the angle between two vectors, ignoring their length:

```
cos(q, d) = (q · d) / (||q|| * ||d||)
```
''' + DIAGRAM_COSINE + r'''

You'll implement it in a vectorised way: `query_vec` is one vector, `matrix` has one chunk vector
per row, and you return one score per row. Guard against divide-by-zero (a zero vector → score 0).

### ✏️ Your turn: implement `cosine_similarity`
''')

exercise(
    stub=r'''
def cosine_similarity(query_vec, matrix):
    """Cosine similarity of `query_vec` against every row of `matrix`.

    `query_vec`: shape (D,).   `matrix`: shape (N, D).   returns: shape (N,) np.ndarray.

    Hints:
      - dot products of every row with the query:  matrix @ query_vec
      - row norms:  np.linalg.norm(matrix, axis=1) ; query norm: np.linalg.norm(query_vec)
      - divide safely so a zero norm yields 0, not NaN (see np.divide's `where=` / `out=`).
    """
    # ✏️ YOUR CODE HERE
    raise NotImplementedError
''',
    solution=r'''
def cosine_similarity(query_vec, matrix):
    """Cosine similarity of `query_vec` against every row of `matrix`."""
    denom = np.linalg.norm(matrix, axis=1) * np.linalg.norm(query_vec)
    scores = matrix @ query_vec
    return np.divide(scores, denom, out=np.zeros_like(scores), where=denom != 0)
''',
)

code(r'''
# _cosine_similarity is a @staticmethod, so wrap it when patching.
RetrievalCore._cosine_similarity = staticmethod(cosine_similarity)
print("Patched RetrievalCore._cosine_similarity")
''')

code(r'''
suite = TestSuite("Part 2.1 — cosine_similarity")

@suite.case("cosine_similarity", "identical direction -> 1.0")
def _():
    q = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    M = np.array([[1, 0, 0], [2, 0, 0]], dtype=np.float32)
    out = cosine_similarity(q, M)
    assert np.allclose(out, [1.0, 1.0])

@suite.case("cosine_similarity", "orthogonal -> 0.0")
def _():
    q = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    M = np.array([[0, 1, 0]], dtype=np.float32)
    assert np.isclose(cosine_similarity(q, M)[0], 0.0)

@suite.case("cosine_similarity", "zero vector is safe (no NaN)")
def _():
    q = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    M = np.array([[0, 0, 0]], dtype=np.float32)
    assert cosine_similarity(q, M)[0] == 0.0

suite.run()
''')

# --- 2.2 Keyword search ---------------------------------------------------- #
md(r'''
## 2.2 — Keyword (lexical) search

Embeddings capture meaning but can miss an exact term — a specific article number, an acronym, a
rare word. **Keyword search** complements them by scoring exact token overlap. We use **BM25**, a
classic ranking function that rewards query terms appearing in a chunk while down-weighting words
that are common across the whole corpus.

The helpers are provided: `_tokenize(text)` (lowercase + split), `self._apply_metadata_filter(...)`,
and `self._top_k(candidates, scores, top_k)`. You wire them together with `BM25Okapi`.

### ✏️ Your turn: implement `keyword_search`
''')

exercise(
    stub=r'''
def keyword_search(self, query, top_k=None, metadata_filter=None):
    """Rank self.chunks by BM25 overlap with `query`. Return list[(Chunk, score)].

    Steps:
      1. top_k = self.top_k if top_k is None else top_k
      2. query_tokens = _tokenize(query)
      3. candidates = self._apply_metadata_filter(self.chunks, metadata_filter)
         If there are no candidates or no query tokens, return [].
      4. corpus = [_tokenize(chunk.text) for chunk in candidates]
         bm25 = BM25Okapi(corpus); scores = bm25.get_scores(query_tokens)
      5. return self._top_k(candidates, scores, top_k)
    """
    # ✏️ YOUR CODE HERE
    raise NotImplementedError
''',
    solution=r'''
def keyword_search(self, query, top_k=None, metadata_filter=None):
    """Rank self.chunks by BM25 overlap with `query`. Return list[(Chunk, score)]."""
    top_k = self.top_k if top_k is None else top_k
    query_tokens = _tokenize(query)
    candidates = self._apply_metadata_filter(self.chunks, metadata_filter)
    if not candidates or not query_tokens:
        return []

    corpus = [_tokenize(chunk.text) for chunk in candidates]
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(query_tokens)
    return self._top_k(candidates, scores, top_k)
''',
)

code(r'''
RetrievalCore.keyword_search = keyword_search
print("Patched RetrievalCore.keyword_search")
''')

code(r'''
suite = TestSuite("Part 2.2 — keyword_search")

_kw_chunks = [
    Chunk(document_id="d", index=0, text="employees get twenty five days of annual leave",
          metadata={"source": "leave.md"}),
    Chunk(document_id="d", index=1, text="expenses must be submitted within thirty days",
          metadata={"source": "expense.md"}),
]
_rc = RetrievalCore(MockEmbedder(), chunks=_kw_chunks)

@suite.case("keyword_search", "the chunk with the query terms ranks first")
def _():
    res = _rc.keyword_search("how many days of annual leave", top_k=2)
    assert res[0][0].index == 0

@suite.case("keyword_search", "top_k caps the number of results")
def _():
    assert len(_rc.keyword_search("days", top_k=1)) == 1

@suite.case("keyword_search", "a metadata filter restricts the candidates")
def _():
    res = _rc.keyword_search("days", metadata_filter={"source": "expense.md"})
    assert all(c.metadata["source"] == "expense.md" for c, _ in res)

@suite.case("keyword_search", "empty query -> no results")
def _():
    assert _rc.keyword_search("   ") == []

suite.run()
''')

# --- 2.3 Hybrid fusion ----------------------------------------------------- #
md(r'''
## 2.3 — Combining the two: Reciprocal Rank Fusion

Keyword and embedding search return scores on **incompatible scales** (BM25 vs. cosine), so you
can't just add them. **Reciprocal Rank Fusion (RRF)** sidesteps this by using only each result's
*rank*:

```
fused_score(chunk) = Σ over lists  1 / (k + rank_in_that_list)        # rank is 0-based, k = 60
```

A chunk that ranks high in *either* list scores well; a chunk near the top of *both* wins. `k=60`
is the conventional dampening constant. Dedupe by `chunk.id` (a chunk can appear in both lists).
''' + DIAGRAM_RRF + r'''

Above: **chunk B** isn't first in either list, but it ranks well in *both*, so fusion floats it to
the top — exactly the behaviour we want.

### ✏️ Your turn: implement `reciprocal_rank_fusion`
''')

exercise(
    stub=r'''
def reciprocal_rank_fusion(ranked_lists, top_k, k=60):
    """Fuse several ranked [(Chunk, score)] lists into one. Return list[(Chunk, fused_score)].

    Steps:
      1. Keep two dicts: fused_scores[chunk_id] -> float, and chunks_by_id[chunk_id] -> Chunk.
      2. For each list, for each (rank, (chunk, _score)) using enumerate (rank is 0-based):
             fused_scores[chunk.id] += 1.0 / (k + rank)
             chunks_by_id[chunk.id] = chunk
      3. Sort chunk ids by fused score descending; return the top_k as (chunk, score) pairs.
    """
    # ✏️ YOUR CODE HERE
    raise NotImplementedError
''',
    solution=r'''
def reciprocal_rank_fusion(ranked_lists, top_k, k=60):
    """Fuse several ranked [(Chunk, score)] lists into one. Return list[(Chunk, fused_score)]."""
    fused_scores = {}
    chunks_by_id = {}
    for ranked in ranked_lists:
        for rank, (chunk, _score) in enumerate(ranked):
            fused_scores[chunk.id] = fused_scores.get(chunk.id, 0.0) + 1.0 / (k + rank)
            chunks_by_id[chunk.id] = chunk

    ordered = sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)
    return [(chunks_by_id[cid], score) for cid, score in ordered[:top_k]]
''',
)

code(r'''
RetrievalCore._reciprocal_rank_fusion = staticmethod(reciprocal_rank_fusion)
print("Patched RetrievalCore._reciprocal_rank_fusion")
''')

code(r'''
suite = TestSuite("Part 2.3 — reciprocal_rank_fusion")

_a = Chunk(document_id="d", index=0, text="A")
_b = Chunk(document_id="d", index=1, text="B")
_c = Chunk(document_id="d", index=2, text="C")

@suite.case("reciprocal_rank_fusion", "a chunk ranked highly in both lists wins")
def _():
    list1 = [(_a, 9.0), (_b, 1.0)]   # b at rank 1
    list2 = [(_b, 9.0), (_c, 1.0)]   # b at rank 0  -> b appears in both
    fused = reciprocal_rank_fusion([list1, list2], top_k=3)
    assert fused[0][0].id == _b.id

@suite.case("reciprocal_rank_fusion", "results are deduplicated by chunk id")
def _():
    fused = reciprocal_rank_fusion([[(_a, 1.0)], [(_a, 1.0)]], top_k=5)
    assert len(fused) == 1

@suite.case("reciprocal_rank_fusion", "top_k limits the output")
def _():
    fused = reciprocal_rank_fusion([[(_a, 1.0), (_b, 1.0), (_c, 1.0)]], top_k=2)
    assert len(fused) == 2

suite.run()
''')

md(r'''
With all three patched in, **hybrid search** (provided, it calls your functions) now works
end-to-end. Let's run a real query over the policy corpus we ingested earlier.
''')

code(r'''
_rc_all = RetrievalCore(MockEmbedder([1.0, 0.0, 0.0]), chunks=all_chunks, top_k=3)
show_results(_rc_all.hybrid_search("how many days of paternity leave"),
             title='Hybrid search · "how many days of paternity leave"')
''')

# --- 2.4 Reranking (bonus) ------------------------------------------------- #
md(r'''
## 2.4 — Reranking *(bonus)*

Fusion gives a good ordering cheaply, but the top results can still be improved by a second, more
careful pass. Production systems use a **cross-encoder** that reads the query and each chunk
*together*. As a lightweight stand-in, implement a reranker that boosts chunks sharing more exact
terms with the query. This is optional and not required by the app.
''')

exercise(
    stub=r'''
def rerank(query, results, top_k=None):
    """Reorder [(Chunk, score)] by exact term overlap with the query (bonus).

    Hint: q_terms = set(_tokenize(query)); score each chunk by
    len(q_terms & set(_tokenize(chunk.text))); sort descending; optionally cut to top_k.
    """
    # ✏️ YOUR CODE HERE (optional)
    raise NotImplementedError
''',
    solution=r'''
def rerank(query, results, top_k=None):
    """Reorder [(Chunk, score)] by exact term overlap with the query (bonus)."""
    q_terms = set(_tokenize(query))

    def overlap(pair):
        return len(q_terms & set(_tokenize(pair[0].text)))

    reranked = sorted(results, key=overlap, reverse=True)
    return reranked if top_k is None else reranked[:top_k]
''',
)

# --- 2.5 LLM answer -------------------------------------------------------- #
md(r'''
## 2.5 — Putting it together: answer with the LLM

The final step of RAG: **retrieve**, stuff the chosen chunk texts into a **context block**, and ask
the LLM to answer *using only that context*. You return both the answer and the source chunks (the
web app shows them as citations).

Provided on `self`: `self._retrieve(query, search_type, metadata_filter)` (dispatches to the right
search), `self.llm.complete(prompt, system_prompt=...)`, and `self.system_prompt`.

### ✏️ Your turn: implement `retrieve_and_answer`
''')

exercise(
    stub=r'''
def retrieve_and_answer(self, query, search_type=None, metadata_filter=None):
    """Retrieve, then answer with the LLM. Return (answer_str, list[(Chunk, score)]).

    Steps:
      1. results = self._retrieve(query, search_type, metadata_filter)
      2. context = "\n\n".join(chunk.text for chunk, _score in results)
      3. prompt  = f"Context:\n{context}\n\nQuestion: {query}"
      4. answer  = self.llm.complete(prompt, system_prompt=self.system_prompt)
      5. return answer, results
    """
    # ✏️ YOUR CODE HERE
    raise NotImplementedError
''',
    solution=r'''
def retrieve_and_answer(self, query, search_type=None, metadata_filter=None):
    """Retrieve, then answer with the LLM. Return (answer_str, list[(Chunk, score)])."""
    results = self._retrieve(query, search_type, metadata_filter)
    context = "\n\n".join(chunk.text for chunk, _score in results)
    prompt = f"Context:\n{context}\n\nQuestion: {query}"
    answer = self.llm.complete(prompt, system_prompt=self.system_prompt)
    return answer, results
''',
)

code(r'''
RAGCore.retrieve_and_answer = retrieve_and_answer
print("Patched RAGCore.retrieve_and_answer")
''')

code(r'''
suite = TestSuite("Part 2.5 — retrieve_and_answer")

# A full RAGCore wired to mocks: no network, fully deterministic.
_rag = RAGCore(MockEmbedder([1.0, 0.0, 0.0]), MockLLM(), chunk_size=5, overlap=1)
_rag.ingest_document(Document(
    text="employees are entitled to twenty five days of annual leave per year",
    metadata={"source": "leave.md", "document_title": "Leave"},
))

@suite.case("retrieve_and_answer", "returns an answer plus the source chunks")
def _():
    answer, sources = _rag.retrieve_and_answer("annual leave days", search_type=SearchType.KEYWORD)
    assert isinstance(answer, str) and answer
    assert len(sources) >= 1

@suite.case("retrieve_and_answer", "the retrieved context reaches the LLM prompt")
def _():
    _rag.retrieve_and_answer("annual leave days", search_type=SearchType.KEYWORD)
    assert "annual leave" in _rag.llm.last_prompt   # MockLLM records what it was sent

suite.run()
''')

md(r'''
**See it rendered.** The mock LLM just echoes the prompt it received, so this view doubles as an
X-ray of *exactly* what your pipeline sends the model — the retrieved context plus the question.
''')

code(r'''
_ans, _src = _rag.retrieve_and_answer("how many days of annual leave", search_type=SearchType.KEYWORD)
show_answer(_ans, _src)
''')

md(r'''
### Live demo *(optional — needs an OpenRouter key)*

If you set a key in 0.2, this ingests the real policy folder and answers a question with the actual
embedding + LLM models. Otherwise it is skipped.
''')

code(r'''
if HAS_KEY:
    from clients.embedder import TextEmbedder
    from clients.llm import LLMClient
    live = RAGCore(TextEmbedder(api_key=os.environ["OPENROUTER_API_KEY"]),
                   LLMClient(api_key=os.environ["OPENROUTER_API_KEY"]))
    live.ingest_path("data")
    answer, sources = live.retrieve_and_answer(
        "Can I accept a CHF 120 gift from a vendor?", search_type=SearchType.HYBRID)
    show_answer(answer, sources)
else:
    print("No API key set — skipping the live demo (the offline tests above already prove it works).")
''')

# --- 2.6 Export ------------------------------------------------------------ #
md(r'''
## 2.6 — Export your Part 2 functions

Writes your retrieval functions to `solutions/part2_retrieval.py`. **Run your implementation cells
above first.**
''')

code(r'''
HEADER_P2 = (
    '"""Part 2 — exported from the notebook. Do not edit by hand; re-export instead."""\n'
    'from __future__ import annotations\n'
    'import numpy as np\n'
    'from rank_bm25 import BM25Okapi\n'
    'from retrieval_core import _tokenize\n\n\n'
)

_funcs_p2 = [cosine_similarity, keyword_search, reciprocal_rank_fusion, retrieve_and_answer]
_body = HEADER_P2 + "\n\n".join(inspect.getsource(f) for f in _funcs_p2)

_path = pathlib.Path("solutions/part2_retrieval.py")
_path.parent.mkdir(exist_ok=True)
_path.write_text(_body)
print(f"Wrote {_path} ({len(_funcs_p2)} functions).")

try:
    from google.colab import files
    files.download(str(_path))
except Exception:
    pass
''')

# =========================================================================== #
# PART 3 — Run the full app
# =========================================================================== #
md(r'''
---
# Part 3 — Run the full application *(optional)*

You've built the engine. The repo also ships a small **web app** around it so you can try your RAG
system in a browser.

## How it fits together

- **Backend** (`server.py`, FastAPI): wraps your pipeline in an HTTP API. It holds one shared
  persistent vector store and builds a fresh `RAGCore` per request. Each request carries *your own*
  OpenRouter key in a header — the server never stores it.
- **Frontend** (`frontend/`, React + Vite): a key field, document upload, a question box with a
  strategy selector, and an answer view with expandable source citations.
- **Your code plugs in via `solutions.apply()`**: on startup the app monkey-patches the two files
  you exported (`solutions/part1_ingestion.py`, `solutions/part2_retrieval.py`) onto the real
  modules — exactly what you did cell-by-cell here. If a file is missing the app falls back to the
  reference implementation, so it always runs.

## Steps

1. Make sure your exported files are in the repo's `solutions/` folder (Parts 1.4 and 2.6).
2. **Backend:**
   ```bash
   uv sync                       # or: pip install -e .
   uv run uvicorn server:app --reload
   ```
3. **Frontend** (in another terminal):
   ```bash
   cd frontend && npm install && npm run dev
   ```
4. Open the printed URL, paste your OpenRouter key, upload the policies in `data/`, and ask away.

> The backend prints `[solutions] applied student implementations: ...` at startup when it picks up
> your exported files — a quick way to confirm your code is the one running.
''')

# =========================================================================== #
# Appendix — your own tests
# =========================================================================== #
md(r'''
---
## Appendix — write your own tests

Use this scratch space to probe anything you're unsure about — print intermediate values, try edge
cases, or add `TestSuite` checks of your own. The same harness the graders use:
''')

code(r'''
suite = TestSuite("My experiments")

@suite.case("sliding_window", "my own edge case")
def _():
    # ✏️ change me
    assert sliding_window("a b c d", chunk_size=2, overlap=0) == ["a b", "c d"]

suite.run()

# ...or just scratch:
# print(chunking.chunk_text(docs[0].text, chunk_size=50, overlap=10)[:2])
''')


# =========================================================================== #
# Write the notebook
# =========================================================================== #
nb = new_notebook(cells=cells)
nb.metadata.update({
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
})

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
with OUTPUT.open("w", encoding="utf-8") as fh:
    nbf.write(nb, fh)

flavour = "SOLUTION" if SOLUTION else "student"
print(f"Wrote {flavour} notebook -> {OUTPUT}  ({len(cells)} cells)")
