from pathlib import Path
from backend.database import update_status
from backend.ingestion.pdf_parser import extract_text_from_pdf, render_pages_to_images
from backend.ingestion.chunker import chunk_pages
from backend.ingestion.text_indexer import build_text_index, build_bm25_index
from backend.utils.file_utils import get_page_image_dir
from backend.utils.logging_config import get_logger

logger = get_logger("manuallens.ingestion.pipeline")

def run_ingestion_sync(document_id: str, pdf_path: str | Path, run_visual: bool = False) -> None:
    """
    Synchronous execution of the document ingestion pipeline.
    Invoked in a worker thread by FastAPI background tasks.
    """
    path_str = str(pdf_path)
    logger.info("Starting ingestion pipeline for document %s (path: %s)", document_id, path_str)

    try:
        # Stage 1: Extract text and check OCR
        update_status(document_id, status="processing", stage="extracting_text")
        pages = extract_text_from_pdf(path_str)
        page_count = len(pages)
        update_status(document_id, status="processing", stage="indexing_text", page_count=page_count)

        # Stage 2: Chunk text and build vector + BM25 index
        chunks = chunk_pages(document_id, pages)
        build_text_index(document_id, chunks)
        build_bm25_index(document_id, chunks)

        # Stage 3: Render pages to PNG images
        update_status(document_id, status="processing", stage="rendering_pages")
        img_dir = get_page_image_dir(document_id)
        image_paths = render_pages_to_images(path_str, output_dir=img_dir, dpi=150)

        # Stage 4: Optional visual indexing (Phase 2)
        if run_visual:
            update_status(document_id, status="processing", stage="indexing_visuals")
            try:
                from backend.ingestion.visual_indexer import build_visual_index
                build_visual_index(document_id, image_paths)
            except Exception as e:
                logger.warning("Visual indexing failed or not configured: %s", e)

        # Stage 5: Ready
        update_status(document_id, status="ready", stage="ready", page_count=page_count)
        logger.info("Ingestion pipeline successfully completed for document %s", document_id)

    except Exception as e:
        logger.error("Ingestion pipeline failed for document %s: %s", document_id, e, exc_info=True)
        update_status(
            document_id,
            status="error",
            stage="error",
            error_message=str(e),
        )
        raise

