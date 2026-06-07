"""src/pipeline.py - Full RAG orchestrator: retrieve → stream generate."""

import sys
from pathlib import Path
from typing import Generator, List, Dict, Tuple

from loguru import logger

sys.path.append(str(Path(__file__).parent.parent))
from config import TOP_K
from src.retriever import retrieve
from src.generator import stream_response, get_no_context_response


def stream_answer(
    query: str,
    k: int = TOP_K,
) -> Tuple[Generator[str, None, None], List[Dict]]:
    """Full RAG pipeline: retrieves relevant chunks, then streams a grounded answer."""
    logger.info(f"RAG pipeline triggered | query='{query[:60]}...' | k={k}")

    sources = retrieve(query, k=k)

    if not sources:
        logger.warning("No relevant chunks found above threshold — returning fallback response")
        return get_no_context_response(), []

    token_stream = stream_response(question=query, chunks=sources)

    return token_stream, sources


if __name__ == "__main__":
    # Standalone test
    TEST_QUERIES = [
        "What personal data is collected from users?",
        "How long is user data retained?",
        "Can users request deletion of their data?",
        "What third parties receive user information?",
        "What are the user's rights under this policy?",
    ]

    print("=" * 70)
    print("RAG PIPELINE — Standalone Test")
    print("=" * 70)

    for i, query in enumerate(TEST_QUERIES, start=1):
        print(f"\n[Query {i}/{len(TEST_QUERIES)}]: {query}")
        print("─" * 60)

        stream, sources = stream_answer(query)

        print("Answer: ", end="", flush=True)
        for token in stream:
            print(token, end="", flush=True)

        print(f"\n\nSources ({len(sources)} chunks):")
        for src in sources:
            print(
                f"  • [{src['chunk_id']}] score={src['score']:.3f} | "
                f"{src['text'][:80].replace(chr(10), ' ')}..."
            )
        print("=" * 70)
