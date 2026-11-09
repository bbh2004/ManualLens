import os
import logging
from pathlib import Path
from dotenv import load_dotenv

logger = logging.getLogger("manuallens.config")

# Project root is the parent directory of backend (i.e. manuallens/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load .env file
env_path = PROJECT_ROOT / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
else:
    load_dotenv()

# External APIs
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")

# Storage paths (resolved relative to PROJECT_ROOT)
STORAGE_DIR: Path = PROJECT_ROOT / os.getenv("STORAGE_DIR", "storage")
PDF_DIR: Path = PROJECT_ROOT / os.getenv("PDF_DIR", "storage/pdfs")
PAGE_IMAGES_DIR: Path = PROJECT_ROOT / os.getenv("PAGE_IMAGES_DIR", "storage/page_images")
SQLITE_DB_PATH: Path = PROJECT_ROOT / os.getenv("SQLITE_DB_PATH", "storage/db/metadata.sqlite")
CHROMA_DIR: Path = PROJECT_ROOT / os.getenv("CHROMA_DIR", "storage/db/chroma")
QDRANT_DIR: Path = PROJECT_ROOT / os.getenv("QDRANT_DIR", "storage/db/qdrant")
BM25_DIR: Path = STORAGE_DIR / "db" / "bm25"

# Ensure runtime directories exist
for path in [PDF_DIR, PAGE_IMAGES_DIR, SQLITE_DB_PATH.parent, CHROMA_DIR, QDRANT_DIR, BM25_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# Processing settings
MAX_PDF_SIZE_MB: int = int(os.getenv("MAX_PDF_SIZE_MB", "50"))
_raw_chunk_size = int(os.getenv("CHUNK_SIZE_TOKENS", "200"))
if _raw_chunk_size > 200:
    logger.warning(
        f"CHUNK_SIZE_TOKENS={_raw_chunk_size} exceeds 200. all-MiniLM-L6-v2 has a 256 token limit. "
        "Enforcing 200 tokens buffer per ADR-011."
    )
    CHUNK_SIZE_TOKENS: int = 200
else:
    CHUNK_SIZE_TOKENS: int = _raw_chunk_size

CHUNK_OVERLAP_TOKENS: int = int(os.getenv("CHUNK_OVERLAP_TOKENS", "20"))
OCR_TEXT_THRESHOLD: int = int(os.getenv("OCR_TEXT_THRESHOLD", "50"))

# Retrieval settings
TEXT_TOP_K: int = int(os.getenv("TEXT_TOP_K", "5"))
VISUAL_TOP_K: int = int(os.getenv("VISUAL_TOP_K", "3"))
RRF_K: int = int(os.getenv("RRF_K", "60"))
ENABLE_VISUAL_INDEXING: bool = os.getenv("ENABLE_VISUAL_INDEXING", "true").lower() in ("true", "1", "yes")

# Modal GPU Acceleration for ColPali
USE_MODAL: bool = os.getenv("USE_MODAL", "true").lower() in ("true", "1", "yes")
MODAL_APP_NAME: str = os.getenv("MODAL_APP_NAME", "manuallens-colpali")
MODAL_GPU: str = os.getenv("MODAL_GPU", "T4")

# LLM settings
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "1500"))
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.1"))

# Server settings
BACKEND_PORT: int = int(os.getenv("BACKEND_PORT", "8000"))
FRONTEND_PORT: int = int(os.getenv("FRONTEND_PORT", "5173"))
raw_cors = os.getenv("CORS_ORIGINS", "http://localhost:5173")
CORS_ORIGINS: list[str] = [origin.strip() for origin in raw_cors.split(",") if origin.strip()]
