import os
import shutil
import uuid
from pathlib import Path
from typing import Tuple
from backend.config import PDF_DIR, PAGE_IMAGES_DIR, BM25_DIR, MAX_PDF_SIZE_MB
from backend.utils.logging_config import get_logger

logger = get_logger("manuallens.file_utils")

def generate_document_id() -> str:
    """Generates an 8-character unique hexadecimal document ID."""
    return uuid.uuid4().hex[:8]

def validate_pdf(file_bytes: bytes, filename: str) -> Tuple[bool, str]:
    """
    Validates that the uploaded payload is a genuine PDF file within the size limit.
    Checks extension, magic bytes, and size.
    """
    if not filename.lower().endswith(".pdf"):
        return False, "File extension must be .pdf"

    max_bytes = MAX_PDF_SIZE_MB * 1024 * 1024
    if len(file_bytes) > max_bytes:
        return False, f"File size exceeds maximum allowed size of {MAX_PDF_SIZE_MB} MB"

    if len(file_bytes) < 4 or not file_bytes.startswith(b"%PDF-"):
        return False, "Invalid PDF file: Missing %PDF- header magic bytes"

    return True, ""

def get_pdf_path(document_id: str, filename: str) -> Path:
    """Returns the persistent storage path for an uploaded PDF."""
    safe_name = Path(filename).name
    return PDF_DIR / f"{document_id}_{safe_name}"

def get_page_image_dir(document_id: str) -> Path:
    """Returns directory path where rendered page PNGs are stored for this document."""
    dir_path = PAGE_IMAGES_DIR / document_id
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path

def cleanup_document_files(document_id: str) -> None:
    """Removes all stored filesystem assets associated with a document_id."""
    # 1. Delete PDF files matching document_id
    for pdf_file in PDF_DIR.glob(f"{document_id}_*"):
        try:
            pdf_file.unlink()
            logger.info("Deleted PDF file: %s", pdf_file)
        except Exception as e:
            logger.warning("Failed to delete PDF %s: %s", pdf_file, e)

    # 2. Delete page images directory
    img_dir = PAGE_IMAGES_DIR / document_id
    if img_dir.exists():
        try:
            shutil.rmtree(img_dir)
            logger.info("Deleted page images directory: %s", img_dir)
        except Exception as e:
            logger.warning("Failed to delete image dir %s: %s", img_dir, e)

    # 3. Delete BM25 pickle
    bm25_file = BM25_DIR / f"{document_id}.pkl"
    if bm25_file.exists():
        try:
            bm25_file.unlink()
            logger.info("Deleted BM25 pickle: %s", bm25_file)
        except Exception as e:
            logger.warning("Failed to delete BM25 file %s: %s", bm25_file, e)

