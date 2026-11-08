from fastapi import APIRouter, UploadFile, File, BackgroundTasks, HTTPException, status
from backend.models import UploadResponse
from backend.utils.file_utils import validate_pdf, generate_document_id, get_pdf_path
from backend.config import ENABLE_VISUAL_INDEXING
from backend.database import create_document
from backend.ingestion.pipeline import run_ingestion_sync
from backend.utils.logging_config import get_logger

logger = get_logger("manuallens.api.upload")
router = APIRouter(prefix="/api", tags=["upload"])

@router.post("/upload", response_model=UploadResponse, status_code=status.HTTP_202_ACCEPTED)
async def upload_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """
    Accepts a single PDF file upload, validates magic bytes and size,
    stores the file, and triggers background ingestion.
    """
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename is missing.",
        )

    content = await file.read()
    is_valid, error_msg = validate_pdf(content, file.filename)
    if not is_valid:
        logger.warning("Upload validation failed for %s: %s", file.filename, error_msg)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error_msg,
        )

    doc_id = generate_document_id()
    dest_path = get_pdf_path(doc_id, file.filename)

    with open(dest_path, "wb") as f:
        f.write(content)

    create_document(
        document_id=doc_id,
        filename=file.filename,
        file_path=str(dest_path),
        file_size_bytes=len(content),
    )

    # Trigger background ingestion thread
    background_tasks.add_task(run_ingestion_sync, doc_id, dest_path, ENABLE_VISUAL_INDEXING)

    logger.info("Accepted PDF upload '%s' -> assigned doc_id: %s", file.filename, doc_id)
    return UploadResponse(
        document_id=doc_id,
        filename=file.filename,
        status="processing",
    )
