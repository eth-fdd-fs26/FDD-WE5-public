"""Tests for retrieval_core.py, grouped by the function each one validates.

Run it:

    uv run python -m tests.test_retrieval_core
    # or
    uv run python tests/test_retrieval_core.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the repo root importable whether run as a module or as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from models import Chunk
from retrieval_core import RetrievalCore
from tests.harness import TestSuite
from tests.mocks import MockEmbedder

suite = TestSuite("Retrieval Core — Test Results")


def _chunk(index: int, text: str, embedding: list[float] | None = None) -> Chunk:
    return Chunk(document_id="doc", index=index, text=text, embedding=embedding)


def _assert_pairs(results) -> None:
    """Every result must be a (Chunk, float) tuple."""
    assert isinstance(results, list), f"expected a list, got {type(results).__name__}"
    for item in results:
        assert isinstance(item, tuple) and len(item) == 2, f"expected (Chunk, score) tuples, got {item!r}"
        chunk, score = item
        assert isinstance(chunk, Chunk), f"first element should be a Chunk, got {type(chunk).__name__}"
        assert isinstance(score, float), f"score should be a float, got {type(score).__name__}"


# --------------------------------------------------------------------------- #
# keyword_search
# --------------------------------------------------------------------------- #
@suite.case("keyword_search", "ranks chunks containing the query term first")
def _():
    chunks = [
        _chunk(0, "the cat sat on the mat"),
        _chunk(1, "dogs are loyal pets"),
        _chunk(2, "a cat is a small domesticated cat animal"),
    ]
    rc = RetrievalCore(MockEmbedder(), chunks)
    order = [c.index for c, _ in rc.keyword_search("cat", top_k=3)]
    assert order[0] == 2, f"chunk 2 mentions 'cat' twice and should rank first; got order {order}"
    assert 1 not in order[:1], f"the non-matching dog chunk should not rank first; got order {order}"


@suite.case("keyword_search", "returns (Chunk, float) tuples")
def _():
    chunks = [_chunk(0, "the cat sat on the mat"), _chunk(1, "dogs are loyal pets")]
    rc = RetrievalCore(MockEmbedder(), chunks)
    _assert_pairs(rc.keyword_search("cat", top_k=2))


@suite.case("keyword_search", "respects top_k")
def _():
    chunks = [_chunk(i, f"cat number {i}") for i in range(5)]
    rc = RetrievalCore(MockEmbedder(), chunks)
    assert len(rc.keyword_search("cat", top_k=2)) == 2, "should return exactly top_k results"


@suite.case("keyword_search", "empty query returns []")
def _():
    rc = RetrievalCore(MockEmbedder(), [_chunk(0, "the cat sat on the mat")])
    assert rc.keyword_search("", top_k=3) == [], "an empty query has no terms to match"


@suite.case("keyword_search", "empty index returns []")
def _():
    rc = RetrievalCore(MockEmbedder(), [])
    assert rc.keyword_search("cat", top_k=3) == [], "no chunks means no results"


# --------------------------------------------------------------------------- #
# embedding_search
# --------------------------------------------------------------------------- #
@suite.case("embedding_search", "ranks the most cosine-similar chunk first")
def _():
    chunks = [
        _chunk(0, "aligned", [1.0, 0.0, 0.0]),
        _chunk(1, "orthogonal", [0.0, 1.0, 0.0]),
        _chunk(2, "close", [0.9, 0.1, 0.0]),
    ]
    rc = RetrievalCore(MockEmbedder([1.0, 0.0, 0.0]), chunks)
    results = rc.embedding_search("q", top_k=3)
    order = [c.index for c, _ in results]
    assert order == [0, 2, 1], f"expected [aligned, close, orthogonal] -> [0, 2, 1], got {order}"
    assert np.isclose(results[0][1], 1.0), f"identical vectors should score ~1.0, got {results[0][1]}"


@suite.case("embedding_search", "skips chunks that have no embedding")
def _():
    chunks = [
        _chunk(0, "embedded", [1.0, 0.0, 0.0]),
        _chunk(1, "not embedded", None),
    ]
    rc = RetrievalCore(MockEmbedder([1.0, 0.0, 0.0]), chunks)
    order = [c.index for c, _ in rc.embedding_search("q", top_k=5)]
    assert order == [0], f"only the embedded chunk should appear; got {order}"


@suite.case("embedding_search", "returns (Chunk, float) tuples")
def _():
    chunks = [_chunk(0, "a", [1.0, 0.0, 0.0]), _chunk(1, "b", [0.0, 1.0, 0.0])]
    rc = RetrievalCore(MockEmbedder([1.0, 0.0, 0.0]), chunks)
    _assert_pairs(rc.embedding_search("q", top_k=2))


@suite.case("embedding_search", "empty index returns []")
def _():
    rc = RetrievalCore(MockEmbedder([1.0, 0.0, 0.0]), [])
    assert rc.embedding_search("q", top_k=3) == [], "no chunks means no results"


# --------------------------------------------------------------------------- #
# _cosine_similarity
# --------------------------------------------------------------------------- #
@suite.case("_cosine_similarity", "scores parallel = 1, orthogonal = 0")
def _():
    query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32)
    scores = RetrievalCore._cosine_similarity(query, matrix)
    assert np.isclose(scores[0], 1.0), f"identical direction should be 1.0, got {scores[0]}"
    assert np.isclose(scores[1], 0.0), f"orthogonal should be 0.0, got {scores[1]}"
    assert np.isclose(scores[2], 1.0), f"same direction (different magnitude) should be 1.0, got {scores[2]}"


@suite.case("_cosine_similarity", "handles zero vectors without dividing by zero")
def _():
    query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    matrix = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    scores = RetrievalCore._cosine_similarity(query, matrix)
    assert scores[0] == 0.0, f"a zero vector should score 0.0, not NaN; got {scores[0]}"


# --------------------------------------------------------------------------- #
# _reciprocal_rank_fusion
# --------------------------------------------------------------------------- #
@suite.case("_reciprocal_rank_fusion", "ranks chunks appearing high in both lists on top")
def _():
    a, b, c = _chunk(0, "a"), _chunk(1, "b"), _chunk(2, "c")
    list1 = [(a, 9.0), (b, 5.0), (c, 1.0)]  # ranks: a, b, c
    list2 = [(b, 0.9), (a, 0.5), (c, 0.1)]  # ranks: b, a, c
    fused = RetrievalCore._reciprocal_rank_fusion([list1, list2], top_k=3)
    order = [chunk.index for chunk, _ in fused]
    assert set(order[:2]) == {0, 1}, f"a and b appear high in both lists; got {order}"
    assert order[2] == 2, f"c is last in both lists and should be last; got {order}"


@suite.case("_reciprocal_rank_fusion", "respects top_k")
def _():
    a, b, c = _chunk(0, "a"), _chunk(1, "b"), _chunk(2, "c")
    fused = RetrievalCore._reciprocal_rank_fusion([[(a, 1.0), (b, 1.0), (c, 1.0)]], top_k=1)
    assert len(fused) == 1, "should return exactly top_k results"


@suite.case("_reciprocal_rank_fusion", "deduplicates a chunk shared across lists")
def _():
    a = _chunk(0, "a")
    fused = RetrievalCore._reciprocal_rank_fusion([[(a, 1.0)], [(a, 1.0)]], top_k=5)
    assert len(fused) == 1, f"the same chunk should appear once, got {len(fused)} entries"


# --------------------------------------------------------------------------- #
# hybrid_search
# --------------------------------------------------------------------------- #
@suite.case("hybrid_search", "returns (Chunk, float) tuples")
def _():
    chunks = [_chunk(0, "the cat sat", [1.0, 0.0, 0.0]), _chunk(1, "loyal dogs", [0.0, 1.0, 0.0])]
    rc = RetrievalCore(MockEmbedder([1.0, 0.0, 0.0]), chunks)
    _assert_pairs(rc.hybrid_search("cat", top_k=2))


@suite.case("hybrid_search", "respects top_k")
def _():
    chunks = [_chunk(i, f"cat {i}", [1.0, 0.0, 0.0]) for i in range(5)]
    rc = RetrievalCore(MockEmbedder([1.0, 0.0, 0.0]), chunks)
    assert len(rc.hybrid_search("cat", top_k=3)) == 3, "should return exactly top_k results"


@suite.case("hybrid_search", "surfaces a chunk strong in both retrievers")
def _():
    # Chunk 0 wins on keywords (mentions 'cat') and on embeddings (aligned vector).
    chunks = [
        _chunk(0, "cat cat cat", [1.0, 0.0, 0.0]),
        _chunk(1, "unrelated text", [0.0, 1.0, 0.0]),
        _chunk(2, "more unrelated", [0.0, 0.0, 1.0]),
    ]
    rc = RetrievalCore(MockEmbedder([1.0, 0.0, 0.0]), chunks)
    order = [c.index for c, _ in rc.hybrid_search("cat", top_k=3)]
    assert order[0] == 0, f"chunk 0 is top in both keyword and embedding search; got {order}"


@suite.case("hybrid_search", "over-retrieval lets a chunk decent in both branches surface")
def _():
    # Chunk 9 is mid-pack in *both* rankings: never top-3 on keywords alone, never
    # top-3 on embeddings alone, but consistently good in both. With a narrow pool
    # (overretrieve=1) neither branch reports it, so fusion cannot see it at all.
    chunks = []
    for i in range(20):
        # More repetitions of 'cat' -> higher BM25; embedding aligned to the query
        # in decreasing order, so the two rankings disagree except in the middle.
        text = " ".join(["cat"] * (20 - i)) + f" filler {i}"
        chunks.append(_chunk(i, text, [1.0, float(i) / 20.0, 0.0]))
    rc = RetrievalCore(MockEmbedder([1.0, 0.0, 0.0]), chunks, top_k=3)

    narrow_pool = {c.id for c, _ in rc.keyword_search("cat", top_k=3)}
    narrow_pool |= {c.id for c, _ in rc.embedding_search("cat", top_k=3)}
    wide_pool = {c.id for c, _ in rc.keyword_search("cat", top_k=60)}
    wide_pool |= {c.id for c, _ in rc.embedding_search("cat", top_k=60)}
    assert len(wide_pool) > len(narrow_pool), (
        f"over-retrieval should widen the fusion pool; got {len(wide_pool)} vs {len(narrow_pool)}"
    )


@suite.case("hybrid_search", "overretrieve=1 reproduces the narrow, pre-fix behaviour")
def _():
    chunks = [_chunk(i, f"cat number {i}", [1.0, float(i) / 10.0, 0.0]) for i in range(10)]
    rc = RetrievalCore(MockEmbedder([1.0, 0.0, 0.0]), chunks, top_k=3)
    got = rc.hybrid_search("cat", overretrieve=1)
    expected = RetrievalCore._reciprocal_rank_fusion(
        [rc.keyword_search("cat", top_k=3), rc.embedding_search("cat", top_k=3)],
        top_k=3,
        k=rc.rrf_k,
    )
    assert [c.id for c, _ in got] == [c.id for c, _ in expected], "overretrieve=1 should fuse only top_k per branch"


@suite.case("hybrid_search", "a pool larger than the corpus degrades gracefully")
def _():
    chunks = [_chunk(0, "the cat sat", [1.0, 0.0, 0.0]), _chunk(1, "loyal dogs", [0.0, 1.0, 0.0])]
    rc = RetrievalCore(MockEmbedder([1.0, 0.0, 0.0]), chunks, top_k=5, overretrieve=50)
    assert len(rc.hybrid_search("cat")) == 2, "should return every chunk, not raise"


@suite.case("hybrid_search", "metadata_filter is applied in both branches")
def _():
    chunks = [_chunk(i, f"cat {i}", [1.0, 0.0, 0.0]) for i in range(6)]
    for i, chunk in enumerate(chunks):
        chunk.metadata = {"source": "a.md" if i % 2 == 0 else "b.md"}
    rc = RetrievalCore(MockEmbedder([1.0, 0.0, 0.0]), chunks, top_k=4)
    results = rc.hybrid_search("cat", metadata_filter={"source": "a.md"})
    assert results, "filter should not empty the results"
    assert all(c.metadata["source"] == "a.md" for c, _ in results), "filtered-out chunks leaked through"


# --------------------------------------------------------------------------- #
# _bm25_index (caching)
# --------------------------------------------------------------------------- #
@suite.case("_bm25_index", "the full-corpus index is built once and reused")
def _():
    rc = RetrievalCore(MockEmbedder(), [_chunk(0, "the cat sat"), _chunk(1, "loyal dogs")])
    first = rc._bm25_index(rc.chunks)
    assert rc._bm25_index(rc.chunks) is first, "the cached index should be reused, not rebuilt"


@suite.case("_bm25_index", "replacing the corpus invalidates the cache")
def _():
    rc = RetrievalCore(MockEmbedder(), [_chunk(0, "the cat sat")])
    stale = rc._bm25_index(rc.chunks)
    rc.chunks = [_chunk(0, "the cat sat"), _chunk(1, "loyal dogs")]
    assert rc._bm25_index(rc.chunks) is not stale, "a new corpus must not reuse the old index"


@suite.case("_bm25_index", "a filtered candidate set gets its own index, not the cached one")
def _():
    chunks = [_chunk(0, "the cat sat"), _chunk(1, "loyal dogs")]
    for i, chunk in enumerate(chunks):
        chunk.metadata = {"source": "a.md" if i == 0 else "b.md"}
    rc = RetrievalCore(MockEmbedder(), chunks)
    cached = rc._bm25_index(rc.chunks)
    narrowed = rc._bm25_index(RetrievalCore._apply_metadata_filter(chunks, {"source": "a.md"}))
    assert narrowed is not cached, "a narrower corpus has different IDF and must be rebuilt"


@suite.case("_bm25_index", "caching does not change what keyword_search returns")
def _():
    chunks = [_chunk(i, f"cat number {i}") for i in range(6)]
    rc = RetrievalCore(MockEmbedder(), chunks, top_k=3)
    first = [(c.id, round(s, 6)) for c, s in rc.keyword_search("cat number 2")]
    second = [(c.id, round(s, 6)) for c, s in rc.keyword_search("cat number 2")]
    assert first == second, "a cached index must return identical scores"


if __name__ == "__main__":
    raise SystemExit(0 if suite.run() else 1)
