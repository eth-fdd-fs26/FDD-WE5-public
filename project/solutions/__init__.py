"""Student-solution loader — binds exported notebook functions onto the app.

The course notebook (``notebook/compliance_rag_project.ipynb``) has participants implement
the key pipeline functions themselves. At the end of each part an *export* cell writes a
file here:

    solutions/part1_ingestion.py   (chunking, ingestion pipeline, metadata filtering)
    solutions/part2_retrieval.py   (cosine, keyword, RRF, the LLM context step)

``apply()`` rebinds each of those functions onto the real project classes/modules, so the
runnable app (``main.py`` / ``server.py``) answers questions using *your* code.

**The repo ships no implementation of these eight functions.** Their bodies in
``chunking.py`` / ``ingestion_core.py`` / ``retrieval_core.py`` / ``rag_core.py`` call
``not_implemented()`` and raise. Nothing works end to end until you export your own — that
is the point of the project, not a bug. No reference implementation is distributed with this
repo; the notebook's per-exercise test cells are how you check your work.

Why the rebinding works: every call site resolves its target through the module global or
the class at call-time (e.g. ``chunk_text`` calls the module-global ``sliding_window``;
``embedding_search`` calls ``self._cosine_similarity``). Rebinding the attribute therefore
flows transparently through the untouched scaffolding — no edits to the cores required.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Callable, NoReturn


def todo(fn: Callable) -> Callable:
    """Mark a shipped stub as not-yet-implemented so ``apply()`` skips it.

    Students never use this decorator — their exported functions are plain. It exists only
    so the placeholder ``part1_ingestion.py`` / ``part2_retrieval.py`` in a fresh checkout
    are recognised as "not done yet" rather than bound onto the app.
    """
    fn._is_todo = True  # type: ignore[attr-defined]
    return fn


@dataclass(frozen=True)
class PatchSpec:
    """One monkey-patch: take ``part_attr`` from the part module, bind it onto a target."""

    part_attr: str  # function name in the part file
    section: str  # the notebook exercise that builds it, e.g. "1.1"
    target_module: str  # module that holds the target
    target_path: str  # dotted path within the module ("name" or "Class.method")
    kind: str  # "function" | "method" | "static"


# The single source of truth for what the two exported files patch.
PART1_SPECS: list[PatchSpec] = [
    PatchSpec("sliding_window", "1.1", "chunking", "sliding_window", "function"),
    PatchSpec("ingest_document", "1.2", "ingestion_core", "IngestionCore.ingest_document", "method"),
    PatchSpec("apply_metadata_filter", "1.3", "retrieval_core", "RetrievalCore._apply_metadata_filter", "static"),
]

PART2_SPECS: list[PatchSpec] = [
    PatchSpec("cosine_similarity", "2.1", "retrieval_core", "RetrievalCore._cosine_similarity", "static"),
    PatchSpec("keyword_search", "2.2", "retrieval_core", "RetrievalCore.keyword_search", "method"),
    PatchSpec("reciprocal_rank_fusion", "2.3", "retrieval_core", "RetrievalCore._reciprocal_rank_fusion", "static"),
    PatchSpec("hybrid_search", "2.4", "retrieval_core", "RetrievalCore.hybrid_search", "method"),
    PatchSpec("retrieve_and_answer", "2.6", "rag_core", "RAGCore.retrieve_and_answer", "method"),
]

# Part files in display order, with a human label for the dashboard.
_PART_FILES = [
    ("Part 1 · Ingestion", "solutions.part1_ingestion", PART1_SPECS),
    ("Part 2 · Retrieval", "solutions.part2_retrieval", PART2_SPECS),
]

_EXPORT_FILE = {
    "solutions.part1_ingestion": "solutions/part1_ingestion.py",
    "solutions.part2_retrieval": "solutions/part2_retrieval.py",
}

_SPEC_BY_NAME = {spec.part_attr: spec for spec in PART1_SPECS + PART2_SPECS}
_FILE_BY_NAME = {
    spec.part_attr: _EXPORT_FILE[modname]
    for _label, modname, specs in _PART_FILES
    for spec in specs
}


def not_implemented(name: str) -> NoReturn:
    """Raise the "this one is yours to write" error for exercise ``name``.

    Called by the eight gutted bodies in the core modules. Keeping the message in one
    place means it always names the right notebook section and the right export file.
    """
    spec = _SPEC_BY_NAME[name]
    raise NotImplementedError(
        f"{name}() is not implemented — it is exercise {spec.section} of the course "
        f"notebook (notebook/compliance_rag_project.ipynb). Write it there, run that "
        f"part's export cell, and put the generated {_FILE_BY_NAME[name]} in place. "
        f"The repo ships no implementation of this function."
    )


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
            continue  # missing or still a placeholder — leave the target raising.
        _bind(spec, fn)
        patched.append(f"{spec.target_module}.{spec.target_path}")
    return patched


def apply_part1() -> list[str]:
    """Patch the Part 1 (ingestion) functions if ``solutions/part1_ingestion.py`` provides them."""
    return _apply_specs("solutions.part1_ingestion", PART1_SPECS)


def apply_part2() -> list[str]:
    """Patch the Part 2 (retrieval) functions if ``solutions/part2_retrieval.py`` provides them."""
    return _apply_specs("solutions.part2_retrieval", PART2_SPECS)


def all_targets() -> list[str]:
    """Every target the two part files must patch for the app to work."""
    return [f"{s.target_module}.{s.target_path}" for s in PART1_SPECS + PART2_SPECS]


def _describe(fn: Callable | None) -> str:
    """First docstring line of a (placeholder or student) function — used as its description."""
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
                    "section": spec.section,
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
        "implemented": done,
        "total": len(functions),
        "complete": done == len(functions),
        "functions": functions,
    }


def apply(verbose: bool = True) -> list[str]:
    """Bind the exported student implementations onto the app. Call once at startup.

    Returns the list of patched targets. Raises ``RuntimeError`` if any of the eight
    functions is still missing — there is no fallback to fall back to, so a caller that
    swallows this error must expect ``NotImplementedError`` at the first query instead.
    """
    patched = apply_part1() + apply_part2()
    missing = sorted(set(all_targets()) - set(patched))

    if verbose:
        if patched:
            print(f"[solutions] applied your implementations: {', '.join(patched)}")
        else:
            print("[solutions] no implementations found in solutions/.")

    if missing:
        raise RuntimeError(
            "These functions are not provided by your solutions/ files (still a "
            "placeholder stub or missing): " + ", ".join(missing) + ".\n"
            "Implement them in the notebook and run the export cells (Parts 1.4 / 2.7). "
            "The repo ships no reference implementation to fall back on."
        )
    return patched
