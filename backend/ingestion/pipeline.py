from pathlib import Path
from backend.config import ENABLE_GRAPHRAG, GRAPH_BUILD_ON_INGEST
from backend.database import update_status
from backend.ingestion.pdf_parser import extract_text_from_pdf, render_pages_to_images
from backend.ingestion.chunker import chunk_pages
from backend.ingestion.text_indexer import build_text_index, build_bm25_index
from backend.utils.file_utils import get_page_image_dir
from backend.utils.logging_config import get_logger

logger = get_logger("manuallens.ingestion.pipeline")

def run_ingestion_sync(
    document_id: str,
    pdf_path: str | Path,
    run_visual: bool = False,
    run_graph: bool = (ENABLE_GRAPHRAG and GRAPH_BUILD_ON_INGEST),
) -> None:
    """
    Synchronous execution of the document ingestion pipeline.
    Invoked in a worker thread by FastAPI background tasks.
    Stages:
    1. extracting_text
    2. indexing_text
    3. indexing_graph (if run_graph=True)
    4. rendering_pages
    5. indexing_visuals (if run_visual=True)
    6. ready
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

        # Stage 3: GraphRAG Knowledge Graph building (Part 2 Feature 1)
        if run_graph:
            update_status(document_id, status="processing", stage="indexing_graph")
            try:
                from backend.knowledge_graph.graph_builder import build_graph_for_document
                build_graph_for_document(document_id, chunks)
            except Exception as e:
                logger.warning("Graph building failed or skipped for doc %s: %s", document_id, e)

        # Stage 4: Render pages to PNG images
        update_status(document_id, status="processing", stage="rendering_pages")
        img_dir = get_page_image_dir(document_id)
        image_paths = render_pages_to_images(path_str, output_dir=img_dir, dpi=150)

        # Stage 5: Optional visual indexing (Phase 2)
        if run_visual:
            update_status(document_id, status="processing", stage="indexing_visuals")
            try:
                from backend.ingestion.visual_indexer import build_visual_index
                build_visual_index(document_id, image_paths)
            except Exception as e:
                logger.warning("Visual indexing failed or not configured: %s", e)

        # Stage 6: Ready
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

