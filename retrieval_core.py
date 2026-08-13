import re

import numpy as np
from rank_bm25 import BM25Okapi

from clients.embedder import TextEmbedder
from models import Chunk, ScoreNormalization, SearchType


# Common English words that carry no discriminating signal for compliance
# document retrieval. Applied to both query and corpus tokens so BM25 IDF
# is computed only over content-bearing terms.
# "not" / "no" are included: they appear in every policy chunk ("must not",
# "not reimbursable", "no special handling") and do not help discriminate
# which document answers a question — the substantive noun does that work.
_STOPWORDS: frozenset[str] = frozenset({
    # articles
    "a", "an", "the",
    # coordinating / subordinating conjunctions
    "and", "or", "but", "if", "because", "although", "though", "since",
    "so", "yet", "nor", "while", "whether",
    # prepositions
    "in", "on", "at", "to", "for", "of", "with", "by", "from", "as",
    "into", "through", "during", "about", "above", "below", "between",
    "after", "before", "until", "against", "without", "within", "under",
    "over", "along",
    # personal pronouns
    "i", "me", "my", "myself", "we", "our", "ours",
    "you", "your", "yourself", "he", "him", "his",
    "she", "her", "it", "its", "they", "them", "their",
    # demonstrative / relative pronouns
    "this", "that", "these", "those", "who", "whom", "which",
    # auxiliary verbs
    "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "must", "shall", "can",
    # negations (ubiquitous in policy language — not discriminating)
    "not", "no",
    # high-frequency adjectives / adverbs
    "all", "any", "each", "few", "more", "most", "other", "some", "such",
    "only", "same", "than", "too", "very", "just", "also", "both",
    # question words (appear in queries, not in answer chunks)
    "what", "when", "where", "how", "why",
    # transition words
    "however", "therefore", "thus",
    # contraction fragments produced by \w+ splitting ("it's" → "it","s")
    "s", "t",
})


def _tokenize(text: str) -> list[str]:
    tokens = re.findall(r"\w+", text.lower())
    return [t for t in tokens if t not in _STOPWORDS]


def _bm25_to_unit(score: float) -> float:
    """Map a raw BM25 score to [0, 1) using s / (1 + s).

    BM25 IDF is negative when a term appears in more than half the corpus,
    which makes the overall score negative. Those are clamped to 0 first
    (negative = no positive signal for this chunk).
    """
    s = max(0.0, score)
    return s / (1.0 + s)


class RetrievalCore:
    def __init__(
        self,
        embedder: TextEmbedder,
        chunks: list[Chunk] | None = None,
        top_k: int = 5,
        rrf_k: int = 60,
        normalization: ScoreNormalization = ScoreNormalization.NORMALIZED,
    ):
        self.embedder = embedder
        # In-memory chunk index. Placeholder until a real vector store is added;
        # ingestion will populate this with embedded chunks.
        self.chunks: list[Chunk] = chunks if chunks is not None else []
        # Defaults the search methods fall back to when not given an explicit
        # value. top_k = how many results to return; rrf_k = the Reciprocal Rank
        # Fusion dampening constant (60 is the canonical value).
        self.top_k = top_k
        self.rrf_k = rrf_k
        # Controls how scores are reported. NORMALIZED applies a fixed transform
        # per method so all three strategies share the same [0, 1] reference:
        #   BM25:   s / (1 + s)  after clamping negatives to 0
        #   cosine: clipped to [0, 1]
        #   RRF:    divided by the theoretical max (n_rankers / k)
        # RAW returns the native values for inspection or debugging.
        self.normalization = normalization

    def set_chunks(self, chunks: list[Chunk]) -> None:
        """Replace the in-memory chunk index searched by all retrieval methods."""
        self.chunks = chunks

    def _keyword_search_raw(
        self,
        query: str,
        top_k: int,
        metadata_filter: dict | None = None,
    ) -> list[tuple[Chunk, float]]:
        """BM25 search — returns raw (un-normalized) scores."""
        query_tokens = _tokenize(query)
        candidates = self._apply_metadata_filter(self.chunks, metadata_filter)
        if not candidates or not query_tokens:
            return []
        # BM25 index is built per-call over the in-memory chunks. Fine for the
        # placeholder store; a real backend would precompute/cache this index.
        corpus = [_tokenize(chunk.text) for chunk in candidates]
        bm25 = BM25Okapi(corpus)
        scores = bm25.get_scores(query_tokens)  # np.ndarray, one score per chunk
        return self._top_k(candidates, scores, top_k)

    def _embedding_search_raw(
        self,
        query: str,
        top_k: int,
        metadata_filter: dict | None = None,
    ) -> list[tuple[Chunk, float]]:
        """Cosine similarity search — returns raw (un-normalized) scores."""
        # Only chunks that have been embedded can be searched semantically.
        candidates = [
            chunk
            for chunk in self._apply_metadata_filter(self.chunks, metadata_filter)
            if chunk.embedding is not None
        ]
        if not candidates:
            return []
        query_vec = np.asarray(self.embedder.embed(query), dtype=np.float32)
        matrix = np.asarray([chunk.embedding for chunk in candidates], dtype=np.float32)
        scores = self._cosine_similarity(query_vec, matrix)
        return self._top_k(candidates, scores, top_k)

    def keyword_search(
        self,
        query: str,
        top_k: int | None = None,
        metadata_filter: dict | None = None,
    ) -> list[tuple[Chunk, float]]:
        top_k = self.top_k if top_k is None else top_k
        results = self._keyword_search_raw(query, top_k, metadata_filter)
        if self.normalization == ScoreNormalization.NORMALIZED:
            # BM25 IDF goes negative when a term appears in more than half the
            # corpus, so scores can be negative. Clamp to 0 first (negative =
            # no useful signal), then apply x/(1+x) which maps [0,∞) → [0,1).
            results = [(chunk, _bm25_to_unit(s)) for chunk, s in results]
        return results

    def embedding_search(
        self,
        query: str,
        top_k: int | None = None,
        metadata_filter: dict | None = None,
    ) -> list[tuple[Chunk, float]]:
        top_k = self.top_k if top_k is None else top_k
        results = self._embedding_search_raw(query, top_k, metadata_filter)
        if self.normalization == ScoreNormalization.NORMALIZED:
            # Cosine is already on [-1, 1]; clip to [0, 1]. Near-orthogonal
            # vectors can produce tiny negatives — treat those as 0 (no match).
            results = [(chunk, max(0.0, min(1.0, s))) for chunk, s in results]
        return results

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
        # Cosine similarity = dot product divided by the product of norms.
        denom = np.linalg.norm(matrix, axis=1) * np.linalg.norm(query_vec)
        scores = matrix @ query_vec
        return np.divide(scores, denom, out=np.zeros_like(scores), where=denom != 0)

    def hybrid_search(
        self,
        query: str,
        top_k: int | None = None,
        metadata_filter: dict | None = None,
    ) -> list[tuple[Chunk, float]]:
        top_k = self.top_k if top_k is None else top_k
        # RRF is rank-based, not score-based — raw sub-scores are sufficient.
        # Normalizing sub-results would be wasted computation.
        keyword_results = self._keyword_search_raw(query, top_k, metadata_filter)
        embedding_results = self._embedding_search_raw(query, top_k, metadata_filter)
        results = self._reciprocal_rank_fusion(
            [keyword_results, embedding_results], top_k=top_k, k=self.rrf_k
        )
        if self.normalization == ScoreNormalization.NORMALIZED:
            # Divide by the theoretical maximum RRF score: a chunk ranked 1st in
            # every ranker earns n_rankers/k. With 2 rankers and k=60 that is
            # 2/60 ≈ 0.033. Dividing puts scores on [0, 1] on a fixed scale —
            # 0.5 means top in one branch but absent from the other; 1.0 means
            # top in both.
            n_rankers = 2
            theoretical_max = n_rankers / self.rrf_k
            results = [(chunk, min(1.0, s / theoretical_max)) for chunk, s in results]
        return results

    @staticmethod
    def _apply_metadata_filter(
        chunks: list[Chunk],
        metadata_filter: dict | None,
    ) -> list[Chunk]:
        """Keep only chunks whose metadata matches every key=value in the filter.

        Metadata filtering makes retrieval more precise: narrow the search to,
        say, ``{"source": "hr_policy.md"}`` or ``{"document_type": "regulation"}``
        before scoring, so unrelated documents can't crowd out the right answer.
        ``None`` (the default) means "no filter — search everything".
        """
        if not metadata_filter:
            return chunks
        return [
            chunk
            for chunk in chunks
            if all(chunk.metadata.get(key) == value for key, value in metadata_filter.items())
        ]

    @staticmethod
    def _reciprocal_rank_fusion(
        ranked_lists: list[list[tuple[Chunk, float]]],
        top_k: int,
        k: int = 60,
    ) -> list[tuple[Chunk, float]]:
        # RRF: each list contributes 1 / (k + rank) per chunk (rank is 0-based).
        # Rank-based fusion avoids normalizing the incompatible score scales of
        # BM25 vs cosine similarity. k=60 is the canonical dampening constant.
        fused_scores: dict[str, float] = {}
        chunks_by_id: dict[str, Chunk] = {}
        for ranked in ranked_lists:
            for rank, (chunk, _score) in enumerate(ranked):
                fused_scores[chunk.id] = fused_scores.get(chunk.id, 0.0) + 1.0 / (k + rank)
                chunks_by_id[chunk.id] = chunk

        ordered = sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)
        return [(chunks_by_id[chunk_id], score) for chunk_id, score in ordered[:top_k]]