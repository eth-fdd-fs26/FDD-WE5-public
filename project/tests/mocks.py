"""Lightweight test doubles so tests run offline (no API keys, no network)."""

from __future__ import annotations

import re
import zlib


class MockEmbedder:
    """Returns a fixed query vector, letting tests control cosine similarity.

    The real ``TextEmbedder`` would call OpenRouter; here we just hand back a
    known vector so similarity scores are fully deterministic.
    """

    def __init__(self, vector: list[float] | None = None):
        self.vector = vector if vector is not None else [1.0, 0.0, 0.0]

    def embed(self, text: str) -> list[float]:
        return list(self.vector)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        # One vector per input text, order preserved — mirrors the real
        # TextEmbedder.embed_batch so ingestion can run offline in tests.
        return [list(self.vector) for _ in texts]


class HashingEmbedder:
    """A deterministic, offline stand-in for a real embedding model.

    ``MockEmbedder`` returns the *same* vector for every text, which is exactly
    what unit tests want (fully controlled cosine scores) and exactly what demos
    do **not**: with identical vectors every chunk ties, so "semantic ranking"
    is really just argpartition's tie-break order.

    This embedder hashes each word into a fixed-width vector and normalises it,
    so texts that share vocabulary land near each other and rankings are
    meaningful. It is **not** a semantic model — it scores word overlap, not
    meaning, so it cannot bridge a paraphrase the way a trained embedder does.
    Use it to make offline demos and evaluations honest, not to draw conclusions
    about how good dense retrieval is.

    ``zlib.crc32`` (not ``hash``) keeps it reproducible across processes; Python
    randomises string hashing per interpreter run.
    """

    def __init__(self, dim: int = 128):
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for token in re.findall(r"\w+", text.lower()):
            vector[zlib.crc32(token.encode()) % self.dim] += 1.0
        norm = sum(value * value for value in vector) ** 0.5
        return [value / norm for value in vector] if norm else vector

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]


class MockLLM:
    """Echoes the prompt back so tests can assert on what was sent."""

    def __init__(self):
        self.last_prompt: str | None = None
        self.last_system_prompt: str | None = None

    def complete(self, prompt: str, system_prompt: str | None = None) -> str:
        self.last_prompt = prompt
        self.last_system_prompt = system_prompt
        return f"[mock-answer] {prompt}"
