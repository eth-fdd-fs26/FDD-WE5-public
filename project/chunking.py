"""Chunking strategies — how a document's text is split before embedding.

Pure functions over strings (no embedder, no database), so the logic stays easy
to read, test, and swap. ``sliding_window`` is the workhorse; ``whole_document``
is the deliberate "no chunking" baseline to compare it against.

The tradeoff, in one line: chunks too small lose surrounding context, chunks too
large produce vaguer vectors and noisier matches. Which size is right is an
empirical question — measure it (see ``evaluation.py``) rather than guess.
"""

from __future__ import annotations

import solutions


def chunk_text(
    text: str,
    *,
    chunk_size: int = 200,
    overlap: int = 40,
    strategy: str = "sliding_window",
) -> list[str]:
    """Split ``text`` into chunks using the chosen strategy.

    Args:
        text: the full document text.
        chunk_size: target chunk length, in words (sliding window only).
        overlap: how many words consecutive chunks share, in words. Overlap keeps
            a sentence that straddles a boundary from being cut off in both
            chunks. Must be smaller than ``chunk_size``.
        strategy: ``"sliding_window"`` (default) or ``"whole_document"``.

    Returns:
        A list of chunk strings (never includes empty chunks).
    """
    if strategy == "sliding_window":
        return sliding_window(text, chunk_size=chunk_size, overlap=overlap)
    if strategy == "whole_document":
        return whole_document(text)
    raise ValueError(
        f"Unknown chunking strategy '{strategy}'. "
        "Use 'sliding_window' or 'whole_document'."
    )


def sliding_window(text: str, chunk_size: int = 200, overlap: int = 40) -> list[str]:
    """Slide a fixed-size window over the words, stepping by ``chunk_size - overlap``.

    Word-based (rather than character-based) so chunks never split a word in half
    and the sizes line up with how people read.

    Example (chunk_size=4, overlap=1) on 9 words:
        words:  [w0 w1 w2 w3] [w3 w4 w5 w6] [w6 w7 w8]
        step = chunk_size - overlap = 3, so windows start at 0, 3, 6.

    Must raise ``ValueError`` for ``chunk_size <= 0``, for ``overlap < 0``, and for
    ``overlap >= chunk_size`` — a window that never advances is an infinite loop,
    not a bad result.
    """
    solutions.not_implemented("sliding_window")


def whole_document(text: str) -> list[str]:
    """The "no chunking" baseline: the entire document as a single chunk.

    Useful for demonstrating *why* chunking is needed — run a retrieval query
    with this strategy and compare the results against ``sliding_window``.
    """
    stripped = text.strip()
    return [stripped] if stripped else []
