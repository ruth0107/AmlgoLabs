"""src/reranker.py - Cross-encoder reranker for second-pass scoring of retrieved chunks."""

import sys
from pathlib import Path
from typing import List, Dict, Optional

from loguru import logger
from sentence_transformers import CrossEncoder

sys.path.append(str(Path(__file__).parent.parent))
from config import RERANKER_MODEL, RERANK_TOP_N

# Singleton
_reranker: Optional[CrossEncoder] = None


def get_reranker() -> CrossEncoder:
    """Lazy-load the cross-encoder model. Cached after first call."""
    global _reranker
    if _reranker is None:
        logger.info(f"Loading reranker model: {RERANKER_MODEL}")
        _reranker = CrossEncoder(RERANKER_MODEL)
        logger.success(f"Reranker ready: {RERANKER_MODEL}")
    return _reranker


# Core rerank function

def rerank(query: str, chunks: List[Dict], top_n: int = RERANK_TOP_N) -> List[Dict]:
    """Reranks the retrieved chunks using a cross-encoder for better accuracy."""
    if not chunks:
        return chunks

    reranker = get_reranker()

    # Build (query, passage) pairs for the cross-encoder
    pairs = [(query, str(chunk["text"])) for chunk in chunks]

    # Predict returns raw logits (higher = more relevant)
    # pyrefly: ignore [no-matching-overload]
    raw_scores: List[float] = reranker.predict(pairs).tolist()

    # Attach rerank_score to each chunk (keep original cosine 'score' intact)
    for chunk, rerank_score in zip(chunks, raw_scores):
        chunk["rerank_score"] = round(rerank_score, 4)

    # Sort descending by rerank_score
    reranked = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)

    logger.info(
        f"Reranked {len(chunks)} candidates → top {top_n} | "
        f"scores: {[r['rerank_score'] for r in reranked[:top_n]]}"
    )

    return reranked[:top_n]


# Smoke test

if __name__ == "__main__":
    sample_query = "What personal data is collected?"
    sample_chunks = [
        {"text": "We collect your name, email, and usage data.", "chunk_id": "chunk_0001", "score": 0.82},
        {"text": "Our refund policy requires a receipt within 30 days.", "chunk_id": "chunk_0002", "score": 0.78},
        {"text": "Personal information such as IP addresses may be logged.", "chunk_id": "chunk_0003", "score": 0.74},
        {"text": "You can contact support at help@example.com.", "chunk_id": "chunk_0004", "score": 0.71},
    ]

    print(f"Query: {sample_query}\n")
    print("Before reranking (by cosine similarity):")
    for c in sample_chunks:
        print(f"  [{c['score']:.3f}] {c['chunk_id']}: {c['text']}")

    results = rerank(sample_query, sample_chunks, top_n=3)

    print("\nAfter reranking (by cross-encoder score):")
    for r in results:
        print(f"  [{r['rerank_score']:.3f}] {r['chunk_id']}: {r['text']}")
