"""Part 1 — Document ingestion pipeline (student export target).

This file is the export target for **Part 1** of the course notebook. The notebook's export
cell overwrites it with your own implementations of the functions below. Until then these are
stubs marked ``@todo``, which the loader (``solutions.apply()``) deliberately skips so the app
keeps running on the reference implementations.

Each function is rebound by ``solutions`` onto the real project code:
    sliding_window   -> chunking.sliding_window
    ingest_document  -> IngestionCore.ingest_document
    derive_title     -> loaders._derive_title

When you export from the notebook, your functions arrive here *without* the ``@todo`` marker,
so the loader patches them in.
"""

from __future__ import annotations

from solutions import todo


@todo
def sliding_window(text: str, chunk_size: int = 200, overlap: int = 40) -> list[str]:
    """Split ``text`` into overlapping word-windows. Implemented in Part 1.1 of the notebook."""
    raise NotImplementedError("Implement sliding_window in the notebook and export it here.")


@todo
def ingest_document(self, document):  # noqa: ANN001 - bound as IngestionCore.ingest_document
    """Run load->chunk->embed->store for one Document. Implemented in Part 1.2 of the notebook."""
    raise NotImplementedError("Implement ingest_document in the notebook and export it here.")


@todo
def derive_title(text: str, stem: str) -> str:
    """First Markdown ``# heading`` else a title-cased stem. Implemented in Part 1.3."""
    raise NotImplementedError("Implement derive_title in the notebook and export it here.")
