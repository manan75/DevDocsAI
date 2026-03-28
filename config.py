"""
config.py — Centralised configuration for DevDocs AI.
All tuneable parameters live here so the rest of the codebase imports from one place.
"""

import os
from pathlib import Path
from dotenv import load_dotenv  
# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
VECTOR_DB_DIR = DATA_DIR / "vector_db"
UPLOAD_DIR = DATA_DIR / "uploads"

DATA_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ─── Ingestion ────────────────────────────────────────────────────────────────
ALLOWED_EXTENSIONS = {
    # Python
    ".py",
    # JavaScript / TypeScript
    ".js", ".ts", ".jsx", ".tsx", ".mjs", ".cjs",
    # PHP
    ".php", ".php3", ".php4", ".php5", ".phtml",
    # Java / Kotlin
    ".java", ".kt", ".kts",
    # C / C++
    ".c", ".cpp", ".h", ".hpp", ".cc",
    # Systems
    ".go", ".rs",
    # Ruby
    ".rb", ".rake",
    # C# / .NET
    ".cs",
    # Shell
    ".sh", ".bash", ".zsh",
    # Docs / Config
    ".md", ".txt", ".yaml", ".yml", ".toml", ".json",
    # HTML / CSS (if you want frontend code)
    ".html", ".css", ".scss",
    # SQL
    ".sql",
}
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
OPENAI_API_KEY = load_dotenv(dotenv_path=Path(__file__).parent / ".env")

# ─── Evaluation ───────────────────────────────────────────────────────────────
JUDGE_MODEL = "openai/gpt-4.1-nano"
EVAL_TOP_K = 5