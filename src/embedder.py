"""src/embedder.py - Singleton wrapper for BAAI/bge-small-en-v1.5 via sentence-transformers."""

import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer

sys.path.append(str(Path(__file__).parent.parent))
from config import EMBEDDING_MODEL

# Singleton model instance (loaded once, reused)
_model: Optional[SentenceTransformer] = None


def get_embedding_model() -> SentenceTransformer:
    """Lazy-load the embedding model. Cached after first load."""
    global _model
    if _model is None:
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        _model = SentenceTransformer(EMBEDDING_MODEL)
        logger.success(f"Embedding model ready | dim={_model.get_embedding_dimension()}")
    return _model


def embed_texts(texts: List[str], batch_size: int = 64) -> List[List[float]]:
    """Embeds a list of document texts in batches."""
    model = get_embedding_model()
    logger.info(f"Embedding {len(texts)} texts in batches of {batch_size}...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,  # L2 normalize for cosine similarity
        convert_to_numpy=True,
    )
    logger.success(f"Done embedding | shape={embeddings.shape}")
    return embeddings.tolist()


def embed_query(query: str) -> List[float]:
    """Embeds a single user query."""
    model = get_embedding_model()
    # BGE models benefit from a query prefix
    prefixed_query = f"Represent this sentence for searching relevant passages: {query}"
    embedding = model.encode(
        [prefixed_query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return embedding[0].tolist()


if __name__ == "__main__":
    sample_texts = [
        "This is the privacy policy for our service.",
        "We collect user data for analytics purposes.",
        "You have the right to delete your personal information.",
    ]
    embeddings = embed_texts(sample_texts)
    print(f"Embedded {len(embeddings)} texts | dim={len(embeddings[0])}")

    query_emb = embed_query("What data do you collect?")
    print(f"Query embedding dim={len(query_emb)}")
