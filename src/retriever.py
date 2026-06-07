"""src/retriever.py - Semantic retrieval from ChromaDB with optional cross-encoder reranking."""

import sys
from pathlib import Path
from typing import List, Dict

from loguru import logger

sys.path.append(str(Path(__file__).parent.parent))
from config import TOP_K, SCORE_THRESHOLD, RETRIEVAL_TOP_K, RERANK_TOP_N, USE_RERANKER
from src.embedder import embed_query
from src.vectorstore import load_collection


def retrieve(query: str, k: int = TOP_K) -> List[Dict]:
    """Retrieves the most semantically relevant chunks for a given query."""
    collection = load_collection()
    query_embedding = embed_query(query)

    # Fetch more candidates when reranking so the cross-encoder has a good pool
    candidate_k = RETRIEVAL_TOP_K if USE_RERANKER else k
    candidate_k = min(candidate_k, collection.count())

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=candidate_k,
        include=["documents", "metadatas", "distances"],
    )

    docs_list = results.get("documents")
    metas_list = results.get("metadatas")
    dist_list = results.get("distances")

    if not docs_list or not metas_list or not dist_list:
        return []

    docs      = docs_list[0]
    metas     = metas_list[0]
    distances = dist_list[0]

    candidates = []
    for doc, meta, dist in zip(docs, metas, distances):
        # ChromaDB cosine distance: 0 = identical, 2 = opposite
        # Convert to similarity: 1 − dist/2  maps [0,2] → [1,0]
        score = round(1.0 - dist / 2.0, 4)

        if score >= SCORE_THRESHOLD:
            candidates.append(
                {
                    "text":       doc,
                    "chunk_id":   meta.get("chunk_id", "unknown"),
                    "source":     meta.get("source", "unknown"),
                    "word_count": meta.get("word_count", 0),
                    "score":      score,
                }
            )

    logger.info(
        f"Vector search: {len(candidates)}/{candidate_k} candidates above "
        f"threshold={SCORE_THRESHOLD} | top cosine={candidates[0]['score'] if candidates else 'n/a'}"
    )

    if not candidates:
        return []

    if USE_RERANKER:
        from src.reranker import rerank  # lazy import to avoid loading model at startup
        retrieved = rerank(query, candidates, top_n=RERANK_TOP_N)
    else:
        # Plain cosine similarity — already sorted by ChromaDB, enforce descending
        candidates.sort(key=lambda x: x["score"], reverse=True)
        retrieved = candidates[:k]

    logger.info(
        f"Final: {len(retrieved)} chunks returned | "
        f"{'rerank' if USE_RERANKER else 'cosine'} scores: "
        f"{[r.get('rerank_score', r['score']) for r in retrieved]}"
    )
    return retrieved


if __name__ == "__main__":
    queries = [
        "What personal data is collected?",
        "How long is data retained?",
        "Can I delete my account?",
    ]
    for q in queries:
        print(f"\nQuery: {q}")
        results = retrieve(q, k=3)
        for r in results:
            primary_score = r.get("rerank_score", r["score"])
            label = "rerank" if "rerank_score" in r else "cosine"
            print(f"  [{label}={primary_score:.3f}] {r['chunk_id']}: {r['text'][:100]}...")
