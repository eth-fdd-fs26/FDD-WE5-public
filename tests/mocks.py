"""Lightweight test doubles so tests run offline (no API keys, no network)."""

from __future__ import annotations


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


class MockLLM:
    """Echoes the prompt back so tests can assert on what was sent."""

    def __init__(self):
        self.last_prompt: str | None = None
        self.last_system_prompt: str | None = None

    def complete(self, prompt: str, system_prompt: str | None = None) -> str:
        self.last_prompt = prompt
        self.last_system_prompt = system_prompt
        return f"[mock-answer] {prompt}"
