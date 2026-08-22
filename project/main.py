"""End-to-end demo of the RAG pipeline.

Ingests the sample documents in ``data/`` and answers a question against them.
This is the first script that exercises the whole pipeline at once:

    load -> chunk -> embed -> store   (ingestion)
    retrieve -> stuff context -> LLM  (query)

Run it (needs an OpenRouter API key for the real embedder + LLM):

    OPENROUTER_API_KEY=sk-... uv run python main.py

The pipeline runs on **your** notebook exports: the eight core functions have no
implementation in this repo, so ``solutions/part1_ingestion.py`` and
``solutions/part2_retrieval.py`` must be in place (Parts 1.4 / 2.7 of the notebook)
before this does anything. Missing ones are named on startup.
"""

import os
import sys

import solutions
from clients.embedder import TextEmbedder
from clients.llm import LLMClient
from rag_core import RAGCore
from retrieval_core import SearchType

DATA_DIR = "data"
SAMPLE_QUERY = "How long is job applicant data kept before it is deleted?"


def main() -> int:
    # Bind the exported notebook solutions onto the pipeline. There is no reference
    # implementation to fall back on, so stop here with a readable message rather
    # than letting the first query raise NotImplementedError from deep in the stack.
    try:
        solutions.apply()
    except RuntimeError as exc:
        print(exc)
        return 1

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("Set OPENROUTER_API_KEY to run the demo, e.g.:")
        print("  OPENROUTER_API_KEY=sk-... uv run python main.py")
        return 1

    embedder = TextEmbedder(api_key=api_key)
    llm = LLMClient(api_key=api_key)
    rag = RAGCore(embedder, llm)

    print(f"Ingesting documents from '{DATA_DIR}/' ...")
    chunks = rag.ingest_path(DATA_DIR)
    print(f"Ingested {len(chunks)} chunks.\n")

    print(f"Question: {SAMPLE_QUERY}")
    answer = rag.receive_query(SAMPLE_QUERY, search_type=SearchType.HYBRID)
    print(f"\nAnswer:\n{answer}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
