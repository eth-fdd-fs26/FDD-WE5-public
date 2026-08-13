"""Chunking strategies — how a document's text is split before embedding.

This is the **Chunking** part of ingestion. It is deliberately kept as a handful
of pure functions over strings (no embedder, no database), so the logic is easy
to read, test, and swap.

Why chunking matters: an embedding model turns a piece of text into a single
vector. If you embed a whole document as one vector, that vector is an average of
many unrelated ideas, and retrieval becomes vague — a question about one
paragraph has to compete with the noise of the entire document. Splitting the
document into smaller, focused chunks gives each idea its own vector, so the
retriever can point at the exact passage that answers a query.

The tradeoff:
  - chunks too small  -> precise match, but each chunk loses surrounding context
  - chunks too large  -> lots of context, but vaguer vectors and noisier matches

`sliding_window` is the workhorse strategy. `whole_document` exists as the "no
chunking" baseline so you can observe, side by side, how much retrieval quality
depends on chunking.
"""

from __future__ import annotations


def chunk_text(
    text: str,
    *,
    chunk_size: int = 100,
    overlap: int = 20,
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
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if overlap < 0:
        raise ValueError(f"overlap must be >= 0, got {overlap}")
    if overlap >= chunk_size:
        raise ValueError(
            f"overlap ({overlap}) must be smaller than chunk_size ({chunk_size}), "
            "otherwise the window would never move forward"
        )

    words = text.split()
    if not words:
        return []

    step = chunk_size - overlap
    chunks: list[str] = []
    for start in range(0, len(words), step):
        window = words[start:start + chunk_size]
        if not window:
            break
        chunks.append(" ".join(window))
        # Once the window reaches the end, stop — otherwise the trailing overlap
        # would emit a final chunk that is fully contained in the previous one.
        if start + chunk_size >= len(words):
            break

    return chunks


def whole_document(text: str) -> list[str]:
    """The "no chunking" baseline: the entire document as a single chunk.

    Useful for demonstrating *why* chunking is needed — run a retrieval query
    with this strategy and compare the results against ``sliding_window``.
    """
    stripped = text.strip()
    return [stripped] if stripped else []
