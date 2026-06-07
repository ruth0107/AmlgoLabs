"""
config.py — Central configuration for the RAG Chatbot.
All tunable parameters live here. Do not hardcode values in other modules.
"""

from pathlib import Path

# Directory Paths
BASE_DIR    = Path(__file__).parent
DATA_DIR    = BASE_DIR / "data"
CHUNKS_DIR  = BASE_DIR / "chunks"
VECTORDB_DIR = BASE_DIR / "vectordb"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"
SRC_DIR     = BASE_DIR / "src"

# Ensure directories exist
for _dir in [DATA_DIR, CHUNKS_DIR, VECTORDB_DIR, NOTEBOOKS_DIR, SRC_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# Document Config
DEFAULT_DOCUMENT = DATA_DIR / "policies.pdf"

# LLM Config (Ollama)
OLLAMA_HOST  = "http://localhost:11434"
LLM_MODEL    = "mistral:7b-instruct"
NUM_CTX      = 2048    # Keep low to fit in 4GB VRAM
TEMPERATURE  = 0.1     # Low = factual, grounded answers
NUM_PREDICT  = 512     # Max tokens to generate

# Embedding Config
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM   = 384

# ChromaDB Config
CHROMA_PATH      = str(VECTORDB_DIR / "chroma_db")
COLLECTION_NAME  = "rag_policy_docs"

# Chunking Config
CHUNK_SIZE_CHARS    = 1000   # ~250 words
CHUNK_OVERLAP_CHARS = 150    # ~40 words overlap
SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", "; ", " ", ""]

# Retrieval Config
TOP_K            = 5     # Number of chunks to retrieve
SCORE_THRESHOLD  = 0.25  # Min cosine similarity to include a chunk
DISPLAY_TOP_K    = 3     # Number of sources to show in the UI

# Reranker Config
# Cross-encoder reranker runs a second-pass scoring over the initial candidates.
# RETRIEVAL_TOP_K: how many candidates to fetch from ChromaDB (cast wide net)
# RERANK_TOP_N:    how many to keep after reranking (= final TOP_K)
# USE_RERANKER:    set False to skip reranking (faster, less accurate)
RERANKER_MODEL   = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RETRIEVAL_TOP_K  = 15    # Broader initial recall for reranker to work with
RERANK_TOP_N     = 5     # Final chunks returned after reranking
USE_RERANKER     = True
