from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct,
)
from typing import Optional
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

    def __init__(self, path: Optional[str] = None):
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

    def create_collection(self, collection_name: str, vector_size: int):
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

    def close(self):
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

    def reset_collection(self, collection_name: str):
        """Drop the collection if it exists (a clean-slate for the caller).

        Backs the "clear my documents" action: the next ingest recreates it.
        """
        if self.client.collection_exists(collection_name):
            self.client.delete_collection(collection_name)
            print(f"Collection '{collection_name}' deleted")

    def ensure_collection(self, collection_name: str, vector_size: int):
        """Create the collection only if it doesn't already exist.

        ``create_collection`` raises if the collection is already there. That is
        fine for a one-shot setup, but ingestion may run repeatedly (and, in
        persistent mode, across restarts), so it needs a create-if-missing guard.

        Args:
            collection_name: name of the collection.
            vector_size: dimension of the embedding vectors.
        """
        if not self.client.collection_exists(collection_name):
            self.create_collection(collection_name, vector_size)

    def insert_points(
        self,
        collection_name: str,
        vectors: list[list[float]],
        payloads: list[dict],
        ids: Optional[list[str]] = None,
    ):
        """
        Insert a batch of embedded chunks into the collection.

        Args:
            collection_name: which collection to insert into
            vectors: list of embedding vectors (one per chunk)
            payloads: list of dicts with text + metadata for each chunk
                      e.g. {"text": "...", "source": "hr_policy.pdf", "department": "HR"}
            ids: optional list of unique IDs. If not provided, UUIDs are generated.
        """
        if len(vectors) != len(payloads):
            raise ValueError(
                f"Mismatch: {len(vectors)} vectors but {len(payloads)} payloads"
            )

        for i, payload in enumerate(payloads):
            errors = validate_payload(payload)
            if errors:
                raise ValueError(f"Invalid payload at index {i}: {errors}")

        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(len(vectors))]

        points = [
            PointStruct(id=point_id, vector=vector, payload=payload)
            for point_id, vector, payload in zip(ids, vectors, payloads)
        ]

        self.client.upsert(collection_name=collection_name, points=points)
        print(f"Inserted {len(points)} points into '{collection_name}'")

    def insert_chunks(self, collection_name: str, chunks: list[Chunk]):
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
            # Same contract as insert_points. Metadata that goes missing here
            # (e.g. an ingestion step that forgets to copy the document's
            # metadata onto its chunks) would otherwise surface much later as
            # inexplicably bad retrieval, or as a metadata filter that silently
            # matches nothing.
            errors = validate_payload(payload)
            if errors:
                raise ValueError(f"Invalid payload for chunk '{chunk.id}': {errors}")
            points.append(PointStruct(id=chunk.id, vector=chunk.embedding, payload=payload))

        self.client.upsert(collection_name=collection_name, points=points)
        print(f"Inserted {len(points)} chunks into '{collection_name}'")

    def insert_document(self, collection_name: str, document: Document, chunks: list[Chunk]):
        """
        Insert a Document together with its Chunks.

        A Document has no embedding of its own, so what actually gets stored are
        its chunks. Each chunk is linked back to the document via document_id and
        inherits the document's metadata (a chunk's own metadata wins on key
        conflicts). The document's chunk_ids list is updated to match what was
        inserted, keeping the in-memory Document consistent with the store.

        Args:
            collection_name: which collection to insert into
            document: the parent Document
            chunks: its chunks — each must have a non-None embedding
        """
        for chunk in chunks:
            chunk.document_id = document.id
            chunk.metadata = {**document.metadata, **chunk.metadata}

        document.chunk_ids = [chunk.id for chunk in chunks]
        self.insert_chunks(collection_name, chunks)

    # NOTE: Qdrant can rank natively (``client.query_points`` with a vector
    # and/or a payload filter), which is what you would use in production. This
    # project deliberately keeps Qdrant as a *persistence layer* only: all
    # ranking happens in RetrievalCore's in-memory implementations, because
    # seeing BM25, cosine and fusion written out is the point of the course.

    def get_all_chunks(self, collection_name: str) -> list[Chunk]:
        """
        Return every chunk stored in the collection as Chunk objects.

        Points come back in Qdrant's internal order, not insertion order, so
        callers that care about document order must sort (``RAGCore._refresh_retrieval``
        sorts by source + document_id + chunk index). Ranked search is unaffected.

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

    def delete(self, collection_name: str, ids: list[str]):
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