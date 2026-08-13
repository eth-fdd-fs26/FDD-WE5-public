from pathlib import Path

from clients.embedder import TextEmbedder
from clients.llm import LLMClient
from db_manager import DBManager
from ingestion_core import IngestionCore
from models import Chunk, Document
from models import ScoreNormalization, SearchType
from retrieval_core import RetrievalCore

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user's question using only the "
    "provided context. If the context does not contain the answer, say so."
)


class RAGCore:
    """Top-level orchestrator tying ingestion, storage, and retrieval together.

    Data flow:
        ingest_*  ->  IngestionCore writes embedded chunks into DBManager (Qdrant)
        _refresh  ->  RetrievalCore is (re)loaded from the store
        answer_query  ->  retrieve -> stuff context -> ask the LLM
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
        normalization: ScoreNormalization = ScoreNormalization.NORMALIZED,
        system_prompt: str = SYSTEM_PROMPT,
        # Ingestion / chunking defaults
        chunk_size: int = 100,
        overlap: int = 25,
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

        # Query defaults used by answer_query (overridable per call).
        self.search_type = search_type
        self.system_prompt = system_prompt

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
        self.retrieval_core = RetrievalCore(self.embedder, top_k=top_k, rrf_k=rrf_k, normalization=normalization)
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
            self.retrieval_core.set_chunks([])
            return
        self.retrieval_core.set_chunks(self.db.get_all_chunks(self.collection_name))

    # ------------------------------------------------------------------ #
    # Query
    # ------------------------------------------------------------------ #
    def _retrieve(
        self,
        query: str,
        search_type: SearchType | None,
        metadata_filter: dict | None,
        top_k: int | None = None,
    ) -> list[tuple[Chunk, float]]:
        """Dispatch to the chosen retrieval strategy (default-aware)."""
        # Fall back to the strategy configured on the orchestrator.
        search_type = self.search_type if search_type is None else search_type
        if search_type == SearchType.KEYWORD:
            return self.retrieval_core.keyword_search(query, top_k=top_k, metadata_filter=metadata_filter)
        if search_type == SearchType.EMBEDDING:
            return self.retrieval_core.embedding_search(query, top_k=top_k, metadata_filter=metadata_filter)
        if search_type == SearchType.HYBRID:
            return self.retrieval_core.hybrid_search(query, top_k=top_k, metadata_filter=metadata_filter)
        raise ValueError(f"Unknown search type: {search_type}")

    def retrieve_and_answer(
        self,
        query: str,
        search_type: SearchType | None = None,
        metadata_filter: dict | None = None,
        top_k: int | None = None,
    ) -> tuple[str, list[tuple[Chunk, float]]]:
        """Retrieve, answer, and return **both** the answer and the chunks used.

        The frontend needs the source chunks (for citations / "show your work"),
        so this is the richer entry point. Use ``answer_query`` instead when
        only the answer string is needed (CLI and notebook use).
        """
        # 1. Retrieve relevant chunks using the chosen strategy.
        results = self._retrieve(query, search_type, metadata_filter, top_k=top_k)

        # 2. Stuff the retrieved chunk texts into a context block.
        context = "\n\n".join(chunk.text for chunk, _score in results)

        # 3. Build the prompt and generate the answer.
        prompt = f"Context:\n{context}\n\nQuestion: {query}"
        answer = self.llm.complete(prompt, system_prompt=self.system_prompt)
        return answer, results

    def answer_query(
        self,
        query: str,
        search_type: SearchType | None = None,
        metadata_filter: dict | None = None,
        top_k: int | None = None,
    ) -> str:
        """Return only the answer string (no source chunks).

        Convenience wrapper around retrieve_and_answer() for CLI and notebook
        use. The server endpoint calls retrieve_and_answer() directly to
        access source citations.
        """
        answer, _results = self.retrieve_and_answer(query, search_type, metadata_filter, top_k=top_k)
        return answer
