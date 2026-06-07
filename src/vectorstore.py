"""src/vectorstore.py - ChromaDB persistent vector store operations."""

import sys
from pathlib import Path
from typing import List, Dict, Optional

import chromadb
from loguru import logger

sys.path.append(str(Path(__file__).parent.parent))
from config import CHROMA_PATH, COLLECTION_NAME
from src.embedder import embed_texts, embed_query

class NoOpEmbeddingFunction:
    def __call__(self, input: List[str]) -> List[List[float]]:
        return []
    
    def embed_with_retries(self, input: List[str], **kwargs) -> List[List[float]]:
        return []

# Batch size for inserting into ChromaDB
INSERT_BATCH_SIZE = 100


def _get_client() -> chromadb.api.ClientAPI:
    """Return a persistent ChromaDB client."""
    return chromadb.PersistentClient(path=CHROMA_PATH)


def build_index(chunks: List[Dict]) -> None:
    """Embeds all chunks and stores them in ChromaDB."""
    client = _get_client()

    # Drop and recreate collection for a clean build
    try:
        client.delete_collection(COLLECTION_NAME)
        logger.info(f"Dropped existing collection: '{COLLECTION_NAME}'")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
        embedding_function=NoOpEmbeddingFunction(),
    )

    texts     = [c["text"]     for c in chunks]
    ids       = [str(c["chunk_id"]) for c in chunks]
    metadatas: List[Dict[str, str | int | float | bool]] = [
        {str(k): v for k, v in c.items() if k != "text"} for c in chunks
    ]

    logger.info(f"Embedding {len(texts)} chunks...")
    embeddings = embed_texts(texts)

    # Insert in batches to avoid memory spikes
    for start in range(0, len(texts), INSERT_BATCH_SIZE):
        end = start + INSERT_BATCH_SIZE
        collection.add(
            documents=texts[start:end],
            embeddings=embeddings[start:end],
            ids=ids[start:end],
            metadatas=metadatas[start:end],
        )

    logger.success(
        f"Index built ✓ | {collection.count()} chunks stored in ChromaDB at '{CHROMA_PATH}'"
    )


def load_collection() -> chromadb.Collection:
    """Load and return the existing ChromaDB collection."""
    client = _get_client()
    collection = client.get_collection(COLLECTION_NAME, embedding_function=NoOpEmbeddingFunction())
    logger.info(f"Collection loaded | {collection.count()} chunks available")
    return collection


def get_chunk_count() -> int:
    """Return the number of indexed chunks (0 if index doesn't exist)."""
    try:
        client = _get_client()
        collection = client.get_collection(COLLECTION_NAME, embedding_function=NoOpEmbeddingFunction())
        return collection.count()
    except Exception:
        return 0


def index_exists() -> bool:
    """Check whether a valid index has already been built."""
    return get_chunk_count() > 0


if __name__ == "__main__":
    from src.document_loader import load_pdf
    from src.chunker import chunk_text, save_chunks

    text   = load_pdf()
    chunks = chunk_text(text)
    save_chunks(chunks)
    build_index(chunks)
    print(f"\nIndex ready: {get_chunk_count()} chunks indexed")
