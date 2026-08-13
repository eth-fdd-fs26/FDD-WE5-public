from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    Range,
)
import uuid

from metadata_schema import validate_payload
from models.chunk import Chunk
from models.document import Document

_CHUNK_STRUCTURAL_KEYS = frozenset({"document_id", "index", "text"})


class DBManager:
    """
    Database Manager for the RAG pipeline.
    Wraps the Qdrant client and exposes clean methods for
    creating collections, inserting chunks, and searching.
    """

    def __init__(self, path: str | None = None):
        """Connect to Qdrant.

        Args:
            path: where to store the data.
                - None (default): in-memory, ephemeral — wiped when the process
                  exits. Good for tests and quick experiments. No Docker needed.
                - a directory path (e.g. "./qdrant_data"): local on-disk storage
                  that persists across restarts, so documents survive between
                  runs and don't need re-ingesting.
        """
        if path is None:
            self.client = QdrantClient(":memory:")
            print("Connected to Qdrant (in-memory mode)")
        else:
            self.client = QdrantClient(path=path)
            print(f"Connected to Qdrant (local persistent mode at '{path}')")

    def create_collection(self, collection_name: str, vector_size: int) -> None:
        """
        Create a collection to store document chunks.

        Args:
            collection_name: name of the collection (e.g. "compliance_documents")
            vector_size: dimension of the embedding vectors (e.g. 384, 1536)
        """
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )
        print(f"Collection '{collection_name}' created (vector size: {vector_size})")

    def close(self) -> None:
        """Release the underlying client (and, in on-disk mode, its file lock).

        Local on-disk Qdrant is single-writer: it holds a lock on the storage
        folder, so the handle must be closed before another process (e.g. a
        reloaded server worker) can open the same path.
        """
        self.client.close()

    def collection_exists(self, collection_name: str) -> bool:
        """Whether a collection has been created yet.

        Useful before reading: a brand-new (e.g. freshly started) store has no
        collection until the first ingest, so callers that want to hydrate from
        the store must guard against scrolling a missing collection.
        """
        return self.client.collection_exists(collection_name)

    def reset_collection(self, collection_name: str) -> None:
        """Drop the collection if it exists (a clean-slate for the caller).

        Backs the "clear my documents" action: the next ingest recreates it.
        """
        if self.collection_exists(collection_name):
            self.client.delete_collection(collection_name)
            print(f"Collection '{collection_name}' deleted")

    def ensure_collection(self, collection_name: str, vector_size: int) -> None:
        """Create the collection only if it doesn't already exist.

        ``create_collection`` raises if the collection is already there. That is
        fine for a one-shot setup, but ingestion may run repeatedly (and, in
        persistent mode, across restarts), so it needs a create-if-missing guard.

        Args:
            collection_name: name of the collection.
            vector_size: dimension of the embedding vectors.
        """
        if not self.collection_exists(collection_name):
            self.create_collection(collection_name, vector_size)

    def insert_chunks(self, collection_name: str, chunks: list[Chunk]) -> None:
        """
        Insert a list of Chunk objects into the collection.

        Args:
            collection_name: which collection to insert into
            chunks: list of Chunk objects — each must have a non-None embedding
        """
        if not chunks:
            return

        points = []
        for chunk in chunks:
            if chunk.embedding is None:
                raise ValueError(f"Chunk '{chunk.id}' has no embedding — embed before inserting")
            payload = {
                "document_id": chunk.document_id,
                "index": chunk.index,
                "text": chunk.text,
                **chunk.metadata,
            }
            points.append(PointStruct(id=chunk.id, vector=chunk.embedding, payload=payload))

        self.client.upsert(collection_name=collection_name, points=points)
        print(f"Inserted {len(points)} chunks into '{collection_name}'")

    def insert_document(self, collection_name: str, document: Document, chunks: list[Chunk]) -> None:
        """
        Insert a Document together with its Chunks.

        A Document has no embedding of its own, so what actually gets stored are
        its chunks. Each chunk must already carry the correct document_id and
        metadata — IngestionCore.ingest_document sets both when it builds the
        chunks, which is the only place chunks are created. The document's
        chunk_ids list is updated to match what was inserted, keeping the
        in-memory Document consistent with the store.

        Args:
            collection_name: which collection to insert into
            document: the parent Document
            chunks: its chunks — each must have document_id, metadata, and a
                non-None embedding already set
        """
        document.chunk_ids = [chunk.id for chunk in chunks]
        self.insert_chunks(collection_name, chunks)

    def get_all_chunks(self, collection_name: str) -> list[Chunk]:
        """
        Return every chunk stored in the collection as Chunk objects.
        returns points in Qdrant's internal order (not insertion order), 
        which is why aml_guidelines appears first in the output. 
        for ordered results, sort by chunk.index after fetching.

        Args:
            collection_name: which collection to read from

        Returns:
            list of Chunk objects with embeddings restored
        """
        chunks: list[Chunk] = []
        offset = None

        while True:
            points, next_offset = self.client.scroll(
                collection_name=collection_name,
                with_vectors=True,
                with_payload=True,
                limit=100,
                offset=offset,
            )
            for point in points:
                payload = point.payload or {}
                metadata = {k: v for k, v in payload.items() if k not in _CHUNK_STRUCTURAL_KEYS}
                chunks.append(Chunk(
                    id=str(point.id),
                    document_id=payload.get("document_id", ""),
                    index=payload.get("index", 0),
                    text=payload.get("text", ""),
                    metadata=metadata,
                    embedding=point.vector if isinstance(point.vector, list) else None,
                ))
            if next_offset is None:
                break
            offset = next_offset

        return chunks

    def list_documents(self, collection_name: str) -> list[dict]:
        """Return one summary entry per unique source document in the collection.

        Groups chunks by their ``source`` metadata key and counts how many chunks
        each document contributed. Returns an empty list when the collection does
        not yet exist (i.e. before the first ingest).

        Args:
            collection_name: which collection to summarise.

        Returns:
            List of dicts, each with keys ``source``, ``title``, and ``chunks``.
        """
        if not self.collection_exists(collection_name):
            return []
        summaries: dict[str, dict] = {}
        for chunk in self.get_all_chunks(collection_name):
            source = chunk.metadata.get("source", "unknown")
            entry = summaries.setdefault(
                source,
                {"source": source, "title": chunk.metadata.get("document_title", source), "chunks": 0},
            )
            entry["chunks"] += 1
        return list(summaries.values())

    # NOTE: only called from tests — not used by the production pipeline (server / main).
    def get_all_documents(self, collection_name: str) -> list[str]:
        """
        Return the unique document IDs of all documents stored in the collection.

        Args:
            collection_name: which collection to read from

        Returns:
            list of unique document_id strings, in insertion order
        """
        seen: set[str] = set()
        document_ids: list[str] = []
        for chunk in self.get_all_chunks(collection_name):
            if chunk.document_id not in seen:
                seen.add(chunk.document_id)
                document_ids.append(chunk.document_id)
        return document_ids

    # NOTE: only called from tests — not used by the production pipeline (server / main).
    def delete(self, collection_name: str, ids: list[str]) -> None:
        """
        Delete specific points by their IDs.

        Args:
            collection_name: which collection to delete from
            ids: list of point IDs to remove
        """
        self.client.delete(
            collection_name=collection_name,
            points_selector=ids,
        )
        print(f"Deleted {len(ids)} points from '{collection_name}'")