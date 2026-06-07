"""src/chunker.py - Sentence-aware text chunking using LangChain."""

import json
import sys
from pathlib import Path
from typing import List, Dict

from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger

sys.path.append(str(Path(__file__).parent.parent))
from config import (
    CHUNKS_DIR,
    CHUNK_SIZE_CHARS,
    CHUNK_OVERLAP_CHARS,
    SEPARATORS,
)


def chunk_text(text: str, source_name: str = "policies.pdf") -> List[Dict]:
    """Splits document text into sentence-aware chunks with metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE_CHARS,
        chunk_overlap=CHUNK_OVERLAP_CHARS,
        separators=SEPARATORS,
        length_function=len,
        is_separator_regex=False,
    )

    raw_chunks = splitter.create_documents([text])

    chunks = []
    for i, chunk in enumerate(raw_chunks):
        content = chunk.page_content.strip()
        if not content:
            continue

        word_count = len(content.split())
        chunks.append(
            {
                "chunk_id": f"chunk_{i:04d}",
                "text": content,
                "source": source_name,
                "word_count": word_count,
                "char_count": len(content),
            }
        )

    total_words = sum(c["word_count"] for c in chunks)
    avg_words = total_words // len(chunks) if chunks else 0
    logger.success(
        f"Chunked into {len(chunks)} segments | avg {avg_words} words | "
        f"range [{min(c['word_count'] for c in chunks)}–{max(c['word_count'] for c in chunks)}] words"
    )
    return chunks


def save_chunks(chunks: List[Dict]) -> Path:
    """Persist chunks to JSON for audit and inspection."""
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = CHUNKS_DIR / "chunks.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved chunks → {output_path}")
    return output_path


def load_chunks() -> List[Dict]:
    """Load previously saved chunks from JSON."""
    chunk_path = CHUNKS_DIR / "chunks.json"
    if not chunk_path.exists():
        raise FileNotFoundError(f"No chunks found at {chunk_path}. Run preprocessing first.")
    with open(chunk_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    logger.info(f"Loaded {len(chunks)} chunks from {chunk_path}")
    return chunks


if __name__ == "__main__":
    from src.document_loader import load_pdf

    text = load_pdf()
    chunks = chunk_text(text)
    save_chunks(chunks)

    print(f"\nSample chunk (chunk_0005):\n{'─'*50}")
    if len(chunks) > 5:
        print(chunks[5]["text"])
        print(f"{'─'*50}")
        print(f"Word count: {chunks[5]['word_count']}")
