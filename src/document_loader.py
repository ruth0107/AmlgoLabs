"""src/document_loader.py - Handles PDF loading and text cleaning."""

import re
import sys
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
from loguru import logger

# Allow running as script or imported as module
sys.path.append(str(Path(__file__).parent.parent))
from config import DEFAULT_DOCUMENT


def load_pdf(filepath: Optional[Path] = None) -> str:
    """Loads and extracts clean text from a PDF file."""
    if filepath is None:
        filepath = DEFAULT_DOCUMENT

    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Document not found: {filepath}")

    logger.info(f"Loading PDF: {filepath.name}")
    doc = fitz.open(str(filepath))

    pages_text = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        cleaned = _clean_page_text(text)
        if cleaned.strip():
            pages_text.append(cleaned)

    doc.close()

    full_text = "\n\n".join(pages_text)
    word_count = len(full_text.split())
    logger.success(
        f"Loaded {len(pages_text)} pages | {len(full_text):,} chars | ~{word_count:,} words"
    )
    return full_text


def _clean_page_text(text: str) -> str:
    """Cleans a single page's extracted text by removing formatting artifacts."""
    # Remove standalone page numbers
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)

    # Remove "Page X of Y" patterns
    text = re.sub(r"[Pp]age\s+\d+\s+of\s+\d+", "", text)

    # Collapse multiple spaces into one
    text = re.sub(r"[ \t]+", " ", text)

    # Collapse excessive newlines (max 2 consecutive)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Strip trailing whitespace from each line
    lines = [line.rstrip() for line in text.split("\n")]
    text = "\n".join(lines)

    return text.strip()


if __name__ == "__main__":
    text = load_pdf()
    print(f"\nFirst 500 characters:\n{'─'*50}")
    print(text[:500])
    print(f"{'─'*50}\n...{len(text)} total characters")
