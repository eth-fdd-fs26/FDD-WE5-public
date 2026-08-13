"""End-to-end demo of the RAG pipeline.

Ingests the sample documents in ``data/`` and answers a question against them.
This is the first script that exercises the whole pipeline at once:

    load -> chunk -> embed -> store   (ingestion)
    retrieve -> stuff context -> LLM  (query)

Run it (needs an OpenRouter API key for the real embedder + LLM):

    OPENROUTER_API_KEY=sk-... uv run python main.py

Choose which implementation runs with ``--impl {auto,reference,student}`` (or the
``RAG_IMPL`` env var); defaults to ``auto`` (student code where exported, else reference):

    OPENROUTER_API_KEY=sk-... uv run python main.py --impl reference
"""

import os
import sys

import solutions
from clients.embedder import TextEmbedder
from clients.llm import LLMClient
from rag_core import RAGCore
from models import SearchType
from dotenv import load_dotenv
load_dotenv()  

DATA_DIR = "data"
SAMPLE_QUERY = "How long is job applicant data kept before it is deleted?"


def _impl_from_argv(argv: list[str]) -> str | None:
    """Read ``--impl <mode>`` / ``--impl=<mode>`` from argv; None if absent."""
    for i, arg in enumerate(argv):
        if arg == "--impl" and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith("--impl="):
            return arg.split("=", 1)[1]
    return None


def main() -> int:
    # Select the implementation to run on: --impl flag > RAG_IMPL env var > "auto".
    # On a fresh checkout "auto"/"student" find only @todo stubs and fall back to the
    # reference code; an exported notebook solution overrides it.
    solutions.apply(mode=_impl_from_argv(sys.argv[1:]))

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
    answer = rag.answer_query(SAMPLE_QUERY, search_type=SearchType.HYBRID)
    print(f"\nAnswer:\n{answer}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
