from fastapi import APIRouter, HTTPException, status
from backend.models import QueryRequest, QueryResponse
from backend.database import get_document
from backend.retrieval.orchestrator import answer_question
from backend.utils.logging_config import get_logger

logger = get_logger("manuallens.api.query")
router = APIRouter(prefix="/api", tags=["query"])

@router.post("/query", response_model=QueryResponse)
def query_document(request: QueryRequest):
    """
    Submits a natural language query for a specific processed manual.
    Runs hybrid retrieval (semantic + BM25) and visual search, then generates
    a grounded response with page-level citations.
    """
    if not request.question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question cannot be empty.",
        )

    doc = get_document(request.document_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{request.document_id}' not found.",
        )

    if doc["status"] != "ready":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Document is not ready for querying (current status: '{doc['status']}').",
        )

    try:
        result = answer_question(request.document_id, request.question)
        return QueryResponse(**result)
    except Exception as e:
        logger.error("Query answering failed for doc %s: %s", request.document_id, e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while answering the question: {str(e)}",
        )

