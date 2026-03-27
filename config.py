"""
config.py — Centralised configuration for DevDocs AI.
All tuneable parameters live here so the rest of the codebase imports from one place.
"""

import os
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
VECTOR_DB_DIR = DATA_DIR / "vector_db"
UPLOAD_DIR = DATA_DIR / "uploads"

DATA_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ─── Ingestion ────────────────────────────────────────────────────────────────
ALLOWED_EXTENSIONS = {".py", ".js", ".ts", ".jsx", ".tsx", ".md", ".txt", ".java", ".go", ".rs", ".cpp", ".c", ".h"}
MAX_FILE_SIZE_MB = 2  # skip files larger than this

# ─── Chunking ─────────────────────────────────────────────────────────────────
CHUNK_SIZE = 400          # tokens (approx characters / 4)
CHUNK_OVERLAP = 60        # token overlap between chunks
CHUNK_SIZE_CHARS = CHUNK_SIZE * 4      # character approximation
CHUNK_OVERLAP_CHARS = CHUNK_OVERLAP * 4

# ─── Embeddings ───────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DEVICE = "cpu"

# ─── Chroma ───────────────────────────────────────────────────────────────────
CHROMA_COLLECTION_NAME = "devdocs"

# ─── Retrieval ────────────────────────────────────────────────────────────────
DEFAULT_TOP_K = 5
DEFAULT_SEARCH_TYPE = "similarity"   # "similarity" | "mmr"
MMR_FETCH_K = 20                     # candidate pool for MMR
MMR_LAMBDA_MULT = 0.5                # diversity vs relevance balance

# ─── LLM ──────────────────────────────────────────────────────────────────────
LLM_MODEL = "openai/gpt-4.1-nano"   # via litellm
LLM_MAX_TOKENS = 1024
LLM_TEMPERATURE = 0.1
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# ─── Evaluation ───────────────────────────────────────────────────────────────
JUDGE_MODEL = "openai/gpt-4.1-nano"
EVAL_TOP_K = 5