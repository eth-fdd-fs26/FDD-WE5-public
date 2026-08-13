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
from models import ScoreNormalization
from retrieval_core import RetrievalCore, _STOPWORDS, _tokenize
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
# _tokenize  (stopword removal)
# --------------------------------------------------------------------------- #
@suite.case("_tokenize", "removes stopwords and keeps content words")
def _():
    tokens = _tokenize("what is the reimbursement rate for personal car use")
    assert "reimbursement" in tokens, "content word 'reimbursement' must survive"
    assert "car" in tokens,          "content word 'car' must survive"
    assert "rate" in tokens,         "content word 'rate' must survive"
    assert "what" not in tokens,     "question word 'what' must be removed"
    assert "is" not in tokens,       "auxiliary 'is' must be removed"
    assert "the" not in tokens,      "article 'the' must be removed"
    assert "for" not in tokens,      "preposition 'for' must be removed"


@suite.case("_tokenize", "stopword removal is consistent between query and corpus")
def _():
    # BM25 requires the same tokenisation for both sides. Verify that the same
    # function is used for query tokens and corpus tokens by checking a term
    # that is a stopword does not appear in either.
    query_tokens = _tokenize("must employees comply with this policy")
    corpus_tokens = _tokenize("employees must comply with all policies")
    assert "must" not in query_tokens,   "'must' should be removed from query"
    assert "must" not in corpus_tokens,  "'must' should be removed from corpus"
    assert "employees" in query_tokens,  "content word must survive in query"
    assert "employees" in corpus_tokens, "content word must survive in corpus"
    assert "comply" in query_tokens and "comply" in corpus_tokens


@suite.case("_tokenize", "empty string and all-stopword input return empty list")
def _():
    assert _tokenize("") == [], "empty string should produce no tokens"
    assert _tokenize("what is the") == [], "all-stopword text should produce no tokens"


@suite.case("_tokenize", "contraction fragments are removed")
def _():
    # \w+ splits "it's" into ["it", "s"]; "s" and "t" are in _STOPWORDS.
    tokens = _tokenize("it's not permitted don't do this")
    assert "s" not in tokens, "contraction fragment 's' must be removed"
    assert "t" not in tokens, "contraction fragment 't' must be removed"
    assert "permitted" in tokens, "content word must survive"


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


# --------------------------------------------------------------------------- #
# score normalization — fixed transforms, not relative to result set
# --------------------------------------------------------------------------- #
@suite.case("keyword_search", "normalized BM25 scores are in [0, 1) via x/(1+x) transform")
def _():
    # "cat" in 2 of 5 chunks → positive IDF → positive BM25; x/(1+x) maps to [0,1).
    chunks = [
        _chunk(0, "the cat sat on the mat"),
        _chunk(1, "a cat is a small domesticated animal"),
        _chunk(2, "dogs are loyal and friendly"),
        _chunk(3, "birds can fly very high"),
        _chunk(4, "fish swim in the ocean"),
    ]
    rc = RetrievalCore(MockEmbedder(), chunks, normalization=ScoreNormalization.NORMALIZED)
    results = rc.keyword_search("cat", top_k=5)
    scores = [s for _, s in results]
    assert all(0.0 <= s < 1.0 for s in scores), f"normalized BM25 must be in [0,1), got {scores}"
    # chunks with "cat" score higher than chunks without it
    scores_by_idx = {c.index: s for c, s in results}
    assert scores_by_idx[0] > scores_by_idx[2], "chunks containing the query term must outscore those that don't"


@suite.case("keyword_search", "RAW mode returns unbounded BM25 scores")
def _():
    # "cat" in only 1 of 5 chunks → high IDF; three occurrences → high TF.
    chunks = [
        _chunk(0, "cat cat cat sitting on a mat"),
        _chunk(1, "dogs are loyal and friendly"),
        _chunk(2, "birds can fly very high"),
        _chunk(3, "fish swim in the deep ocean"),
        _chunk(4, "horses gallop across open fields"),
    ]
    rc = RetrievalCore(MockEmbedder(), chunks, normalization=ScoreNormalization.RAW)
    results = rc.keyword_search("cat", top_k=5)
    scores = [s for _, s in results]
    assert any(s > 1.0 for s in scores), f"raw BM25 for a rare, repeated term should exceed 1.0, got {scores}"


@suite.case("keyword_search", "normalization does not change result ordering")
def _():
    # "cat" in 2 of 5 chunks → positive IDF → unambiguous ranking for top 2.
    # x/(1+x) is strictly increasing so rank order is preserved.
    chunks = [
        _chunk(0, "the cat sat on the mat"),
        _chunk(1, "a cat is a small domesticated cat animal"),
        _chunk(2, "dogs are loyal and friendly"),
        _chunk(3, "birds can fly very high"),
        _chunk(4, "fish swim in the deep ocean"),
    ]
    rc_raw  = RetrievalCore(MockEmbedder(), chunks, normalization=ScoreNormalization.RAW)
    rc_norm = RetrievalCore(MockEmbedder(), chunks, normalization=ScoreNormalization.NORMALIZED)
    # top_k=2 selects only the chunks with positive BM25 (deterministic order)
    raw_order  = [c.index for c, _ in rc_raw.keyword_search("cat",  top_k=2)]
    norm_order = [c.index for c, _ in rc_norm.keyword_search("cat", top_k=2)]
    assert raw_order == norm_order, f"normalization must not change ranking: raw={raw_order}, norm={norm_order}"


@suite.case("embedding_search", "normalized cosine scores are absolute clip, not re-scaled")
def _():
    # With min-max the lowest score would be rescaled to 0; with a simple clip
    # each score reflects true similarity — a score of 0.7 stays 0.7.
    chunks = [
        _chunk(0, "aligned",  [1.0, 0.0, 0.0]),   # cosine = 1.0  with query [1,0,0]
        _chunk(1, "diagonal", [0.7, 0.7, 0.0]),   # cosine ≈ 0.707
        _chunk(2, "close",    [0.9, 0.1, 0.0]),   # cosine ≈ 0.994
    ]
    rc = RetrievalCore(MockEmbedder([1.0, 0.0, 0.0]), chunks, normalization=ScoreNormalization.NORMALIZED)
    results = rc.embedding_search("q", top_k=3)
    scores_by_idx = {c.index: s for c, s in results}
    assert all(0.0 <= s <= 1.0 for s in scores_by_idx.values()), f"scores must be in [0,1], got {scores_by_idx}"
    # chunk 0: cosine=1.0 → stays 1.0
    assert np.isclose(scores_by_idx[0], 1.0), f"identical vectors: expected 1.0, got {scores_by_idx[0]}"
    # chunk 1: cosine≈0.707 → stays ~0.707, NOT rescaled down to 0.0
    assert scores_by_idx[1] > 0.5, f"diagonal chunk should retain ~0.707, got {scores_by_idx[1]}"


@suite.case("embedding_search", "RAW mode returns cosine scores on their native scale")
def _():
    chunks = [
        _chunk(0, "aligned",    [1.0, 0.0, 0.0]),
        _chunk(1, "orthogonal", [0.0, 1.0, 0.0]),
    ]
    rc = RetrievalCore(MockEmbedder([1.0, 0.0, 0.0]), chunks, normalization=ScoreNormalization.RAW)
    results = rc.embedding_search("q", top_k=2)
    scores = {c.index: s for c, s in results}
    assert np.isclose(scores[0], 1.0), f"identical vectors should give raw cosine 1.0, got {scores[0]}"
    assert np.isclose(scores[1], 0.0), f"orthogonal vectors should give raw cosine 0.0, got {scores[1]}"


@suite.case("hybrid_search", "normalized RRF scores are on a fixed [0,1] scale by theoretical max")
def _():
    # chunk 0 is the only one mentioning "cat" (positive IDF) and has the most
    # aligned embedding → ranked 1st in both → RRF = 2/k = theoretical max → 1.0.
    chunks = [
        _chunk(0, "cat cat cat", [1.0, 0.0, 0.0]),
        _chunk(1, "unrelated",   [0.0, 1.0, 0.0]),
        _chunk(2, "other stuff", [0.0, 0.0, 1.0]),
        _chunk(3, "no match",    [0.2, 0.8, 0.0]),
        _chunk(4, "still no",    [0.1, 0.9, 0.0]),
    ]
    rc = RetrievalCore(MockEmbedder([1.0, 0.0, 0.0]), chunks, normalization=ScoreNormalization.NORMALIZED)
    results = rc.hybrid_search("cat", top_k=5)
    scores = [s for _, s in results]
    assert all(0.0 <= s <= 1.0 for s in scores), f"normalized RRF must be in [0,1], got {scores}"
    scores_by_idx = {c.index: s for c, s in results}
    assert np.isclose(scores_by_idx[0], 1.0), f"chunk top in both rankings should score 1.0, got {scores_by_idx[0]}"


@suite.case("hybrid_search", "RAW mode returns small RRF fractions")
def _():
    chunks = [_chunk(i, f"cat {i}", [1.0, 0.0, 0.0]) for i in range(5)]
    rc = RetrievalCore(MockEmbedder([1.0, 0.0, 0.0]), chunks, normalization=ScoreNormalization.RAW)
    results = rc.hybrid_search("cat", top_k=5)
    scores = [s for _, s in results]
    assert all(s < 1.0 for s in scores), f"raw RRF scores should be < 1.0, got {scores}"


if __name__ == "__main__":
    raise SystemExit(0 if suite.run() else 1)
