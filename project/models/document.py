from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4


@dataclass
class Document:
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    chunk_ids: list[str] = field(default_factory=list)
    id: str = field(default_factory=lambda: str(uuid4()))