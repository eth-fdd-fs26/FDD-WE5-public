from pathlib import Path

import solutions
from clients.embedder import TextEmbedder
from clients.llm import LLMClient
from db_manager import DBManager
from ingestion_core import IngestionCore
from models import Chunk, Document
from retrieval_core import RetrievalCore, SearchType

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user's question using only the "
    "provided context. If the context does not contain the answer, say so."
)

# Returned instead of calling the LLM when retrieval comes back empty. A RAG
# system that always answers will happily answer from nothing; knowing when to
# abstain is a feature, and skipping the pointless LLM call is a bonus.
NO_CONTEXT_ANSWER = (
    "I could not find anything relevant to that question in the indexed "
    "documents, so I cannot answer it from the available context."
)


class RAGCore:
    """Top-level orchestrator tying ingestion, storage, and retrieval together.

    Data flow:
        ingest_*  ->  IngestionCore writes embedded chunks into DBManager (Qdrant)
        _refresh  ->  RetrievalCore is (re)loaded from the store
        receive_query  ->  retrieve -> stuff context -> ask the LLM
    """

    def __init__(
        self,
        embedder: TextEmbedder,
        llm: LLMClient,
        db: DBManager | None = None,
        db_path: str | None = None,
        collection_name: str = "documents",
        # Retrieval defaults
        search_type: SearchType = SearchType.HYBRID,
        top_k: int = 5,
        rrf_k: int = 60,
        overretrieve: int = 20,
        min_score: float | None = None,
        system_prompt: str = SYSTEM_PROMPT,
        # Ingestion / chunking defaults
        chunk_size: int = 200,
        overlap: int = 40,
        strategy: str = "sliding_window",
    ):
        self.embedder = embedder
        self.llm = llm
        # Storage backend, in order of precedence:
        #   - pass `db` to reuse/share an existing DBManager;
        #   - else pass `db_path` to build one persisting on disk at that path;
        #   - else (both None) build a fresh in-memory Qdrant (ephemeral).
        if db is not None and db_path is not None:
            raise ValueError("Pass either `db` or `db_path`, not both.")
        self.db = db if db is not None else DBManager(path=db_path)
        self.collection_name = collection_name

        # Query defaults used by receive_query (overridable per call).
        self.search_type = search_type
        self.system_prompt = system_prompt
        # Drop retrieved chunks scoring below this before they reach the LLM.
        # None (default) = keep everything. The meaningful value depends on the
        # strategy, because the scales differ: cosine is in [-1, 1], BM25 is
        # unbounded, and RRF scores cluster near 1/rrf_k. Set it per use case.
        self.min_score = min_score

        # Sub-cores are configured from the orchestrator's parameters, so chunking
        # behaviour and result count can be set in one place at the call site.
        self.ingestion_core = IngestionCore(
            self.embedder,
            self.db,
            collection_name,
            chunk_size=chunk_size,
            overlap=overlap,
            strategy=strategy,
        )
        self.retrieval_core = RetrievalCore(
            self.embedder, top_k=top_k, rrf_k=rrf_k, overretrieve=overretrieve
        )
        # Hydrate immediately so a RAGCore built over an already-populated store
        # (e.g. a persistent DB, or one shared across web requests) is queryable
        # without needing an ingest first.
        self._refresh_retrieval()

    # ------------------------------------------------------------------ #
    # Ingestion entry points — each refreshes the retriever afterwards.
    # ------------------------------------------------------------------ #
    def ingest_document(self, document: Document) -> list[Chunk]:
        chunks = self.ingestion_core.ingest_document(document)
        self._refresh_retrieval()
        return chunks

    def ingest_path(self, path: str | Path) -> list[Chunk]:
        """Ingest a single file or a whole directory of ``.txt``/``.md`` files."""
        path = Path(path)
        if path.is_dir():
            chunks = self.ingestion_core.ingest_directory(path)
        else:
            chunks = self.ingestion_core.ingest_file(path)
        self._refresh_retrieval()
        return chunks

    def _refresh_retrieval(self) -> None:
        """Load the stored chunks back into the in-memory retriever.

        RetrievalCore searches an in-memory list (the "simple vector database").
        We rebuild that list from Qdrant after each ingestion so newly added
        documents become searchable. Guarded so it is a no-op before the first
        ingest (the collection doesn't exist yet).
        """
        if not self.db.collection_exists(self.collection_name):
            self.retrieval_core.chunks = []
            return
        # Qdrant returns points in its own internal order, so sort back into
        # document order. Ranked search results don't care, but anything that
        # reads neighbouring chunks (or just prints the corpus) does.
        self.retrieval_core.chunks = sorted(
            self.db.get_all_chunks(self.collection_name),
            key=lambda chunk: (chunk.metadata.get("source", ""), chunk.document_id, chunk.index),
        )

    # ------------------------------------------------------------------ #
    # Query
    # ------------------------------------------------------------------ #
    def _retrieve(
        self,
        query: str,
        search_type: SearchType | None,
        metadata_filter: dict | None,
    ) -> list[tuple[Chunk, float]]:
        """Dispatch to the chosen retrieval strategy, then apply ``min_score``."""
        # Fall back to the strategy configured on the orchestrator.
        search_type = self.search_type if search_type is None else search_type
        if search_type == SearchType.KEYWORD:
            results = self.retrieval_core.keyword_search(query, metadata_filter=metadata_filter)
        elif search_type == SearchType.EMBEDDING:
            results = self.retrieval_core.embedding_search(query, metadata_filter=metadata_filter)
        elif search_type == SearchType.HYBRID:
            results = self.retrieval_core.hybrid_search(query, metadata_filter=metadata_filter)
        else:
            raise ValueError(f"Unknown search type: {search_type}")

        if self.min_score is None:
            return results
        # Retrieval always returns *something* — the top_k nearest chunks exist
        # even when nothing is actually relevant. The threshold is what turns
        # "here are the least-bad chunks" into "I have nothing for you".
        return [(chunk, score) for chunk, score in results if score >= self.min_score]

    def retrieve_and_answer(
        self,
        query: str,
        search_type: SearchType | None = None,
        metadata_filter: dict | None = None,
    ) -> tuple[str, list[tuple[Chunk, float]]]:
        """Retrieve, answer, and return **both** the answer and the chunks used.

        The frontend needs the source chunks (for citations / "show your work"),
        so this is the richer entry point; ``receive_query`` wraps it when only
        the answer string is wanted.

        Retrieve with ``_retrieve``; if nothing survives, return
        ``NO_CONTEXT_ANSWER`` and no sources *without* calling the LLM. Otherwise
        stuff the retrieved chunk texts into a context block, build the prompt, and
        answer with ``self.llm.complete(prompt, system_prompt=...)``.
        """
        solutions.not_implemented("retrieve_and_answer")

    def receive_query(
        self,
        query: str,
        search_type: SearchType | None = None,
        metadata_filter: dict | None = None,
    ) -> str:
        answer, _results = self.retrieve_and_answer(query, search_type, metadata_filter)
        return answer
