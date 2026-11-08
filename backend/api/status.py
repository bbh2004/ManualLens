from fastapi import APIRouter, HTTPException, status
from backend.models import StatusResponse
from backend.database import get_document
from backend.utils.logging_config import get_logger

logger = get_logger("manuallens.api.status")
router = APIRouter(prefix="/api", tags=["status"])

@router.get("/status/{document_id}", response_model=StatusResponse)
def get_document_status(document_id: str):
    """
    Returns current ingestion status, processing stage, and error details if any.
    Polled by the frontend during document processing.
    """
    doc = get_document(document_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID '{document_id}' not found.",
        )

    return StatusResponse(
        document_id=document_id,
        status=doc["status"],
        stage=doc.get("status_stage"),
        error_message=doc.get("error_message"),
    )

