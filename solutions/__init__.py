"""Student-solution loader — monkey-patches exported notebook functions onto the app.

The course notebook (``notebook/compliance_rag_project.ipynb``) has students implement
the key pipeline functions themselves. At the end of each part an *export* cell writes a
file here:

    solutions/part1_ingestion.py   (chunking, ingestion pipeline, metadata)
    solutions/part2_retrieval.py   (cosine, keyword, RRF, the LLM context step)

``apply()`` rebinds each of those functions onto the real project classes/modules, so the
runnable app (``main.py`` / ``server.py``) answers questions using *the student's* code
instead of the reference implementation shipped in the repo.

Why this works: every call site resolves its target through the module global or the class
at call-time (e.g. ``chunk_text`` calls the module-global ``sliding_window``;
``embedding_search`` calls ``self._cosine_similarity``). Rebinding the attribute therefore
flows transparently through the untouched scaffolding — no edits to the cores required.

It is deliberately *guarded*: a missing file, a missing function, or an un-edited stub (one
still marked ``@todo``) is skipped, so a fresh checkout with no student work keeps running on
the provided reference implementations.
"""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from typing import Callable

# Which implementation the app runs on. Resolved from ``RAG_IMPL`` when not passed
# explicitly to ``apply()``:
#   "auto"      — student code where exported, reference everywhere else (default).
#   "reference" — force the repo's reference implementation; ignore solutions/.
#   "student"   — force student code; error if any function is still a stub/missing.
VALID_MODES = ("auto", "reference", "student")
DEFAULT_MODE = "auto"

# The mode the live app last applied (set by ``apply()``), so ``status()`` can report it.
_active_mode: str | None = None


def todo(fn: Callable) -> Callable:
    """Mark a shipped stub as not-yet-implemented so ``apply()`` skips it.

    Students never use this decorator — their exported functions are plain. It exists only
    so the default ``part1_ingestion.py`` / ``part2_retrieval.py`` stubs do not clobber the
    working app before anyone has done the project.
    """
    fn._is_todo = True  # type: ignore[attr-defined]
    return fn


@dataclass(frozen=True)
class PatchSpec:
    """One monkey-patch: take ``part_attr`` from the part module, bind it onto a target."""

    part_attr: str  # function name in the part file
    target_module: str  # module that holds the target
    target_path: str  # dotted path within the module ("name" or "Class.method")
    kind: str  # "function" | "method" | "static"


# The single source of truth for what the two exported files patch.
PART1_SPECS: list[PatchSpec] = [
    PatchSpec("sliding_window", "chunking", "sliding_window", "function"),
    PatchSpec("ingest_document", "ingestion_core", "IngestionCore.ingest_document", "method"),
    PatchSpec("derive_title", "loaders", "_derive_title", "function"),
]

PART2_SPECS: list[PatchSpec] = [
    PatchSpec("cosine_similarity", "retrieval_core", "RetrievalCore._cosine_similarity", "static"),
    PatchSpec("keyword_search", "retrieval_core", "RetrievalCore.keyword_search", "method"),
    PatchSpec("reciprocal_rank_fusion", "retrieval_core", "RetrievalCore._reciprocal_rank_fusion", "static"),
    PatchSpec("retrieve_and_answer", "rag_core", "RAGCore.retrieve_and_answer", "method"),
]


def _bind(spec: PatchSpec, fn: Callable) -> None:
    """Rebind ``fn`` onto the target described by ``spec``."""
    module = importlib.import_module(spec.target_module)
    parts = spec.target_path.split(".")
    owner = module
    for name in parts[:-1]:  # walk into a class if the path is "Class.method"
        owner = getattr(owner, name)
    value = staticmethod(fn) if spec.kind == "static" else fn
    setattr(owner, parts[-1], value)


def _apply_specs(part_module_name: str, specs: list[PatchSpec]) -> list[str]:
    """Apply every spec whose function exists and is implemented. Returns patched names."""
    try:
        part = importlib.import_module(part_module_name)
    except ModuleNotFoundError:
        return []  # the student hasn't dropped this file in — nothing to do.

    patched: list[str] = []
    for spec in specs:
        fn = getattr(part, spec.part_attr, None)
        if fn is None or getattr(fn, "_is_todo", False):
            continue  # missing or still a stub — leave the reference implementation in place.
        _bind(spec, fn)
        patched.append(f"{spec.target_module}.{spec.target_path}")
    return patched


def apply_part1() -> list[str]:
    """Patch the Part 1 (ingestion) functions if ``solutions/part1_ingestion.py`` provides them."""
    return _apply_specs("solutions.part1_ingestion", PART1_SPECS)


def apply_part2() -> list[str]:
    """Patch the Part 2 (retrieval) functions if ``solutions/part2_retrieval.py`` provides them."""
    return _apply_specs("solutions.part2_retrieval", PART2_SPECS)


def _resolve_mode(mode: str | None) -> str:
    """Pick the implementation mode: explicit arg > ``RAG_IMPL`` env var > default."""
    if mode is None:
        mode = os.environ.get("RAG_IMPL", DEFAULT_MODE)
    mode = mode.strip().lower()
    if mode not in VALID_MODES:
        raise ValueError(f"RAG_IMPL must be one of {VALID_MODES}, got {mode!r}")
    return mode


def all_targets() -> list[str]:
    """Every target the two part files can patch (used to check completeness)."""
    return [f"{s.target_module}.{s.target_path}" for s in PART1_SPECS + PART2_SPECS]


# Part files in display order, with a human label for the dashboard.
_PART_FILES = [
    ("Part 1 · Ingestion", "solutions.part1_ingestion", PART1_SPECS),
    ("Part 2 · Retrieval", "solutions.part2_retrieval", PART2_SPECS),
]


def _describe(fn: Callable | None) -> str:
    """First docstring line of a (stub or student) function — used as its description."""
    return (fn.__doc__ or "").strip().split("\n", 1)[0] if fn is not None else ""


def status() -> dict:
    """Report which exported functions are implemented vs still stubbed/missing.

    Pure introspection (never patches), so it is safe to call on every request.
    Drives the frontend progress dashboard. ``reason`` is ``ok`` | ``stub`` | ``missing``.
    """
    functions: list[dict] = []
    for label, modname, specs in _PART_FILES:
        try:
            part = importlib.import_module(modname)
        except ModuleNotFoundError:
            part = None  # the student hasn't added this file yet.
        for spec in specs:
            fn = getattr(part, spec.part_attr, None) if part is not None else None
            if fn is None:
                reason = "missing"
            elif getattr(fn, "_is_todo", False):
                reason = "stub"
            else:
                reason = "ok"
            functions.append(
                {
                    "part": label,
                    "name": spec.part_attr,
                    "target": f"{spec.target_module}.{spec.target_path}",
                    "kind": spec.kind,
                    "implemented": reason == "ok",
                    "reason": reason,
                    "description": _describe(fn),
                }
            )
    done = sum(f["implemented"] for f in functions)
    return {
        "mode": _active_mode if _active_mode is not None else _resolve_mode(None),
        "implemented": done,
        "total": len(functions),
        "complete": done == len(functions),
        "functions": functions,
    }


def apply(verbose: bool = True, mode: str | None = None) -> list[str]:
    """Select the implementation the live app runs on. Safe to call once at startup.

    ``mode`` (or the ``RAG_IMPL`` env var when ``mode is None``) chooses between
    ``"auto"``, ``"reference"`` and ``"student"`` — see ``VALID_MODES``. Returns the
    list of patched targets (empty when running on the reference implementation).
    """
    global _active_mode
    mode = _resolve_mode(mode)
    _active_mode = mode

    if mode == "reference":
        if verbose:
            print("[solutions] mode=reference — using the repo's reference implementation.")
        return []

    patched = apply_part1() + apply_part2()

    if mode == "student":
        missing = sorted(set(all_targets()) - set(patched))
        if missing:
            raise RuntimeError(
                "RAG_IMPL=student, but these functions are not provided by your "
                "solutions/ files (still a @todo stub or missing): " + ", ".join(missing)
                + ".\nExport them from the notebook (Parts 1.4 / 2.6), or use RAG_IMPL=auto."
            )

    if verbose:
        if patched:
            print(f"[solutions] mode={mode} — applied student implementations: {', '.join(patched)}")
        else:
            print(f"[solutions] mode={mode} — no student implementations found; using reference.")
    return patched
