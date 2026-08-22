import re
from enum import Enum

import numpy as np
from rank_bm25 import BM25Okapi

import solutions
from clients.embedder import TextEmbedder
from models import Chunk


class SearchType(str, Enum):
    KEYWORD = "keyword"
    EMBEDDING = "embedding"
    HYBRID = "hybrid"


def _tokenize(text: str) -> list[str]:
    # Lowercase + split on word boundaries. No stopword removal for now;
    # BM25's IDF term already down-weights common words.
    return re.findall(r"\w+", text.lower())


class RetrievalCore:
    def __init__(
        self,
        embedder: TextEmbedder,
        chunks: list[Chunk] | None = None,
        top_k: int = 5,
        rrf_k: int = 60,
        overretrieve: int = 20,
    ):
        self.embedder = embedder
        # Cached BM25 index over the full corpus, invalidated by the `chunks`
        # setter below. Must exist before `self.chunks` is assigned.
        self._bm25: BM25Okapi | None = None
        # In-memory chunk index. Placeholder until a real vector store is added;
        # ingestion will populate this with embedded chunks.
        self.chunks = chunks if chunks is not None else []
        # Defaults the search methods fall back to when not given an explicit
        # value. top_k = how many results to return; rrf_k = the Reciprocal Rank
        # Fusion dampening constant (60 is the canonical value).
        self.top_k = top_k
        self.rrf_k = rrf_k
        # How wide each branch of hybrid search casts its net before fusion:
        # each retriever returns top_k * overretrieve candidates. Fusion can only
        # rerank what it is given, so a pool of just top_k per branch leaves RRF
        # nothing to work with. overretrieve=1 is the old, narrow behaviour.
        self.overretrieve = overretrieve

    # ------------------------------------------------------------------ #
    # The chunk index, and the BM25 index built over it
    # ------------------------------------------------------------------ #
    @property
    def chunks(self) -> list[Chunk]:
        """The in-memory corpus this retriever searches."""
        return self._chunks

    @chunks.setter
    def chunks(self, value: list[Chunk] | None) -> None:
        # Replacing the corpus invalidates the cached BM25 index. This is the
        # whole reason `chunks` is a property: RAGCore reassigns it after every
        # ingest, and a stale index would silently search the old corpus.
        self._chunks = list(value) if value is not None else []
        self._bm25 = None

    def _bm25_index(self, candidates: list[Chunk]) -> BM25Okapi:
        """A BM25 index over ``candidates``, cached when they are the full corpus.

        BM25 has to see the whole corpus to compute its IDF term, so the index
        is rebuilt whenever the candidate set changes. Building it is O(corpus),
        and a naive ``BM25Okapi(corpus)`` inside ``keyword_search`` pays that on
        *every query* — including twice per hybrid query, over a wide
        over-retrieval pool. Caching the unfiltered index turns that into a
        one-off cost per ingest.

        A metadata filter produces a different (narrower) corpus with different
        IDF, so those indexes are built ad hoc and not cached.
        """
        if candidates is self._chunks:
            if self._bm25 is None:
                self._bm25 = BM25Okapi([_tokenize(chunk.text) for chunk in candidates])
            return self._bm25
        return BM25Okapi([_tokenize(chunk.text) for chunk in candidates])

    def keyword_search(
        self,
        query: str,
        top_k: int | None = None,
        metadata_filter: dict | None = None,
    ) -> list[tuple[Chunk, float]]:
        """Rank ``self.chunks`` by BM25 overlap with ``query``.

        Narrow the corpus with ``_apply_metadata_filter`` *before* scoring, bail out
        on an empty corpus or an empty query (BM25 raises on an empty corpus), score
        the query tokens against ``self._bm25_index(candidates)``, and hand the
        scores to ``self._top_k``.
        """
        solutions.not_implemented("keyword_search")

    def embedding_search(
        self,
        query: str,
        top_k: int | None = None,
        metadata_filter: dict | None = None,
    ) -> list[tuple[Chunk, float]]:
        top_k = self.top_k if top_k is None else top_k
        # Only chunks that have been embedded can be searched semantically.
        candidates = [
            chunk
            for chunk in self._apply_metadata_filter(self.chunks, metadata_filter)
            if chunk.embedding is not None
        ]
        if not candidates:
            return []

        # Embedding the query
        query_vec = np.asarray(self.embedder.embed(query), dtype=np.float32)
        matrix = np.asarray([chunk.embedding for chunk in candidates], dtype=np.float32)

        scores = self._cosine_similarity(query_vec, matrix)
        return self._top_k(candidates, scores, top_k)

    @staticmethod
    def _top_k(candidates: list[Chunk], scores: np.ndarray, top_k: int) -> list[tuple[Chunk, float]]:
        if len(candidates) == 0:
            return []
        top_k = min(top_k, len(candidates))
        # argpartition gives the top_k cheaply, then sort just those by score desc.
        top_idx = np.argpartition(scores, -top_k)[-top_k:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
        return [(candidates[i], float(scores[i])) for i in top_idx]

    @staticmethod
    def _cosine_similarity(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        """Cosine similarity of one ``(D,)`` query against every row of an ``(N, D)`` matrix.

        Cosine similarity = dot product divided by the product of the norms. Vectorised:
        one score per row, no Python loop, and a zero vector must score 0.0 rather than
        producing a NaN.
        """
        solutions.not_implemented("cosine_similarity")

    def hybrid_search(
        self,
        query: str,
        top_k: int | None = None,
        metadata_filter: dict | None = None,
        overretrieve: int | None = None,
    ) -> list[tuple[Chunk, float]]:
        """Run both retrievers wide, fuse their rankings, then cut to ``top_k``.

        The **over-retrieval** step is what makes fusion worth doing. If each
        branch only returned ``top_k``, the fuser would see at most ``2 * top_k``
        candidates and a chunk ranked, say, 8th by BM25 and 6th by cosine —
        decent in both, exactly what RRF exists to promote — could never surface,
        because neither list was long enough to contain it. Retrieving
        ``top_k * overretrieve`` per branch gives fusion a real candidate pool.

        It is close to free here: both branches already score *every* chunk
        (BM25 over the whole corpus, cosine over the whole matrix) and then throw
        the rest away in ``_top_k``, so widening the pool only changes how many
        results are kept, not how much work is done.
        """
        solutions.not_implemented("hybrid_search")

    @staticmethod
    def _apply_metadata_filter(
        chunks: list[Chunk],
        metadata_filter: dict | None,
    ) -> list[Chunk]:
        """Keep only chunks whose metadata matches every key=value in the filter.

        Metadata filtering makes retrieval more precise: narrow the search to,
        say, ``{"source": "hr_policy.md"}`` or ``{"document_type": "regulation"}``
        before scoring, so unrelated documents can't crowd out the right answer.
        ``None`` (the default) means "no filter — search everything", in which case
        hand back the *same list object*: ``_bm25_index`` recognises the full corpus by
        identity, so copying it would silently defeat the BM25 cache.
        """
        solutions.not_implemented("apply_metadata_filter")

    @staticmethod
    def _reciprocal_rank_fusion(
        ranked_lists: list[list[tuple[Chunk, float]]],
        top_k: int,
        k: int = 60,
    ) -> list[tuple[Chunk, float]]:
        """Fuse several ranked ``[(Chunk, score)]`` lists into one ranking.

        RRF: each list a chunk appears in contributes ``1 / (k + rank)`` to its total
        (rank is 0-based). Rank-based fusion avoids normalizing the incompatible score
        scales of BM25 vs cosine similarity; k=60 is the canonical dampening constant.
        Deduplicate by ``chunk.id``, sort by fused score descending, cut to ``top_k``.
        """
        solutions.not_implemented("reciprocal_rank_fusion")
