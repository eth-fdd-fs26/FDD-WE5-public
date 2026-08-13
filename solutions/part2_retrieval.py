"""Part 2 — RAG core / retrieval (student export target).

This file is the export target for **Part 2** of the course notebook. The notebook's export
cell overwrites it with your own implementations of the functions below. Until then these are
stubs marked ``@todo``, which the loader (``solutions.apply()``) deliberately skips so the app
keeps running on the reference implementations.

Each function is rebound by ``solutions`` onto the real project code:
    cosine_similarity        -> RetrievalCore._cosine_similarity
    keyword_search           -> RetrievalCore.keyword_search
    reciprocal_rank_fusion   -> RetrievalCore._reciprocal_rank_fusion
    retrieve_and_answer      -> RAGCore.retrieve_and_answer

When you export from the notebook, your functions arrive here *without* the ``@todo`` marker,
so the loader patches them in.
"""

from __future__ import annotations

from solutions import todo


@todo
def cosine_similarity(query_vec, matrix):  # noqa: ANN001 - bound as RetrievalCore._cosine_similarity
    """Cosine similarity of a query vector against a matrix of chunk vectors. Part 2.1."""
    raise NotImplementedError("Implement cosine_similarity in the notebook and export it here.")


@todo
def keyword_search(self, query: str, top_k=None, metadata_filter=None):
    """BM25 keyword search over self.chunks. Implemented in Part 2.2 of the notebook."""
    raise NotImplementedError("Implement keyword_search in the notebook and export it here.")


@todo
def reciprocal_rank_fusion(ranked_lists, top_k: int, k: int = 60):
    """Fuse ranked result lists with Reciprocal Rank Fusion. Implemented in Part 2.3."""
    raise NotImplementedError("Implement reciprocal_rank_fusion in the notebook and export it here.")


@todo
def retrieve_and_answer(self, query: str, search_type=None, metadata_filter=None):
    """Retrieve, stuff context, ask the LLM; return (answer, sources). Implemented in Part 2.5."""
    raise NotImplementedError("Implement retrieve_and_answer in the notebook and export it here.")
