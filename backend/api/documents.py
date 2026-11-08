from fastapi import APIRouter, HTTPException, status
from backend.models import DocumentListResponse, DocumentInfo, DeleteResponse
from backend.database import list_documents, get_document, delete_document
from backend.ingestion.text_indexer import delete_text_index
from backend.utils.file_utils import cleanup_document_files
from backend.utils.logging_config import get_logger

logger = get_logger("manuallens.api.documents")
router = APIRouter(prefix="/api", tags=["documents"])

@router.get("/documents", response_model=DocumentListResponse)
def get_documents():
    """Returns a list of all documents and their processing statuses."""
    docs = list_documents()
    items = [
        DocumentInfo(
            document_id=d["document_id"],
            filename=d["filename"],
            page_count=d.get("page_count"),
            upload_time=d["upload_time"],
            status=d["status"],
        )
        for d in docs
    ]
    return DocumentListResponse(documents=items)

@router.delete("/documents/{document_id}", response_model=DeleteResponse)
def remove_document(document_id: str):
    """
    Deletes a document and all associated data:
    1. SQLite metadata entry
    2. Original PDF file
    3. Rendered page PNG images
    4. ChromaDB text collection
    5. BM25 serialized index
    6. ColPali/Qdrant visual index (if exists)
    """
    doc = get_document(document_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{document_id}' not found.",
        )

    # 1. Clean ChromaDB and BM25 index
    delete_text_index(document_id)

    # 2. Clean visual index if available
    try:
        from backend.ingestion.visual_indexer import delete_visual_index
        delete_visual_index(document_id)
    except Exception as e:
        logger.debug("Visual index cleanup skipped or failed: %s", e)

    # 3. Clean files from disk (PDF, page images)
    cleanup_document_files(document_id)

    # 4. Delete SQLite row
    delete_document(document_id)

    logger.info("Purged all traces of document: %s", document_id)
    return DeleteResponse(message="Document deleted successfully.")

