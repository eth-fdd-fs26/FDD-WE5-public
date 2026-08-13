"""IngestionCore — the document-ingestion pipeline.

Ingestion is the "write" side of a RAG system: it takes raw documents and turns
them into something the retriever can search. The pipeline has four steps, which
line up with the three things you'll study:

    load  ->  CHUNK  ->  EMBED  ->  store

  1. load   — read a file into a ``Document`` (see ``loaders.py``). This is also
              where **metadata** (source, title) gets attached.
  2. chunk  — split the document into focused passages (see ``chunking.py``).
  3. embed  — turn each chunk's text into a vector (the **embeddings** step).
  4. store  — persist the embedded chunks in the vector database (``DBManager``).

After ingestion, ``RetrievalCore`` reads the stored chunks back and searches them
(the "simple vector database" is its in-memory cosine search; Qdrant is the
persistent store + the advanced approximate-nearest-neighbour path).
"""

from __future__ import annotations

from pathlib import Path

import chunking
import loaders
from clients.embedder import TextEmbedder
from db_manager import DBManager
from models import Chunk, Document


class IngestionCore:
    """Loads, chunks, embeds, and stores documents into the vector database."""

    def __init__(
        self,
        embedder: TextEmbedder,
        db: DBManager,
        collection_name: str = "documents",
        chunk_size: int = 100,
        overlap: int = 25,
        strategy: str = "sliding_window",
    ):
        """
        Args:
            embedder: turns chunk text into embedding vectors.
            db: the vector store the chunks are persisted into.
            collection_name: which collection to write to.
            chunk_size / overlap / strategy: passed straight to ``chunking``.
                Try ``strategy="whole_document"`` to feel how much worse
                retrieval gets without chunking.
        """
        self.embedder = embedder
        self.db = db
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.strategy = strategy

    # ------------------------------------------------------------------ #
    # Core pipeline
    # ------------------------------------------------------------------ #
    def ingest_document(self, document: Document) -> list[Chunk]:
        """Run the full pipeline for one already-loaded ``Document``.

        Returns the list of embedded, stored ``Chunk`` objects.
        """
        # 1. CHUNK — split the document text into passages.
        texts = chunking.chunk_text(
            document.text,
            chunk_size=self.chunk_size,
            overlap=self.overlap,
            strategy=self.strategy,
        )
        if not texts:
            return []

        # 2. BUILD — wrap each passage in a Chunk, carrying the document's own
        #    metadata (source, document_title) onto every chunk it produced.
        chunks = [
            Chunk(
                document_id=document.id,
                index=i,
                text=text,
                metadata=dict(document.metadata),
            )
            for i, text in enumerate(texts)
        ]

        # 3. EMBED — one batched call; embed_batch preserves input order.
        embeddings = self.embedder.embed_batch([chunk.text for chunk in chunks])
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding

        # 4. STORE — make sure the collection exists (sized from the actual
        #    embedding dimension), then persist the document + its chunks.
        self.db.ensure_collection(self.collection_name, vector_size=len(embeddings[0]))
        self.db.insert_document(self.collection_name, document, chunks)

        return chunks

    # ------------------------------------------------------------------ #
    # Convenience loaders
    # ------------------------------------------------------------------ #
    def ingest_file(self, path: str | Path) -> list[Chunk]:
        """Load a single ``.txt``/``.md`` file and ingest it."""
        document = loaders.load_document(path)
        return self.ingest_document(document)

    def ingest_directory(self, path: str | Path) -> list[Chunk]:
        """Load and ingest every supported file in a directory.

        Returns every chunk created across all the documents.
        """
        all_chunks: list[Chunk] = []
        for document in loaders.load_directory(path):
            all_chunks.extend(self.ingest_document(document))
        return all_chunks
