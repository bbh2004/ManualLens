from pathlib import Path
from typing import Any
import fitz  # PyMuPDF
from PIL import Image

from backend.config import OCR_TEXT_THRESHOLD
from backend.ingestion.ocr import run_ocr
from backend.utils.logging_config import get_logger

logger = get_logger("manuallens.pdf_parser")

def render_page_to_image(page: fitz.Page, dpi: int = 150) -> Image.Image:
    """Renders a single PyMuPDF page to a PIL Image."""
    pix = page.get_pixmap(dpi=dpi)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    return img

def extract_text_from_pdf(pdf_path: str) -> list[dict[str, Any]]:
    """
    Extracts text page-by-page using PyMuPDF.
    If a page yields fewer than OCR_TEXT_THRESHOLD characters, attempts OCR fallback.
    Returns:
        [{"page_number": 1, "text": "...", "source_type": "text"|"ocr"}, ...]
    """
    logger.info("Extracting text from PDF: %s", pdf_path)
    pages: list[dict[str, Any]] = []

    doc = fitz.open(pdf_path)
    try:
        total_pages = len(doc)
        for i, page in enumerate(doc):
            page_num = i + 1  # 1-indexed page numbers
            text = page.get_text()

            if len(text.strip()) < OCR_TEXT_THRESHOLD:
                logger.debug(
                    "Page %d has short text (%d chars). Running OCR fallback.",
                    page_num,
                    len(text.strip()),
                )
                img = render_page_to_image(page, dpi=150)
                ocr_text = run_ocr(img)
                if len(ocr_text.strip()) > len(text.strip()):
                    text = ocr_text
                    source_type = "ocr"
                else:
                    source_type = "text"
            else:
                source_type = "text"

            pages.append({
                "page_number": page_num,
                "text": text.strip(),
                "source_type": source_type,
                "total_pages": total_pages,
            })
    finally:
        doc.close()

    logger.info("Extracted %d pages from %s", len(pages), pdf_path)
    return pages

def render_pages_to_images(pdf_path: str, output_dir: str | Path, dpi: int = 150) -> list[str]:
    """
    Renders every PDF page as a PNG image at specified DPI.
    Stores files as page_{n}.png under output_dir.
    Returns list of generated image file paths.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    image_paths: list[str] = []

    logger.info("Rendering pages to PNG in %s at %d DPI", out_dir, dpi)
    doc = fitz.open(pdf_path)
    try:
        for i, page in enumerate(doc):
            page_num = i + 1
            img_path = out_dir / f"page_{page_num}.png"
            pix = page.get_pixmap(dpi=dpi)
            pix.save(str(img_path))
            image_paths.append(str(img_path))
    finally:
        doc.close()

    logger.info("Successfully rendered %d page images", len(image_paths))
    return image_paths

