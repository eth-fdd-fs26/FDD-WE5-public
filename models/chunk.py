from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4


@dataclass
class Chunk:
    document_id: str
    index: int
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    # Kept as list[float] so the model stays JSON-serializable and matches the
    # embedder's output. Could switch to a numpy array for faster vector math,
    # but leave it as-is for now.
    embedding: list[float] | None = None
    id: str = field(default_factory=lambda: str(uuid4()))