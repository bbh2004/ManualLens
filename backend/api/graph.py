import pickle
from fastapi import APIRouter, HTTPException, BackgroundTasks, status
from backend.config import BM25_DIR, ENABLE_GRAPHRAG
from backend.database import get_document
from backend.knowledge_graph.graph_store import get_graph_store
from backend.knowledge_graph.graph_builder import build_graph_for_document
from backend.knowledge_graph.graph_querier import query_knowledge_graph
from backend.utils.logging_config import get_logger

logger = get_logger("manuallens.api.graph")
router = APIRouter(prefix="/api/documents", tags=["graph"])


def _load_chunks_for_doc(doc_id: str) -> list[dict]:
    """Helper to load indexed chunks for document from BM25 pickle store."""
    bm25_path = BM25_DIR / f"{doc_id}.pkl"
    if bm25_path.exists():
        try:
            with open(bm25_path, "rb") as f:
                data = pickle.load(f)
            return data.get("chunks", [])
        except Exception as e:
            logger.warning("Could not read BM25 pickle for doc %s: %s", doc_id, e)
    return []


@router.post("/{document_id}/build-graph")
def build_graph_endpoint(document_id: str, background_tasks: BackgroundTasks, async_mode: bool = False):
    """
    Triggers entity extraction and knowledge graph construction for document_id.
    Can run synchronously or asynchronously via FastAPI background tasks.
    """
    doc = get_document(document_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{document_id}' not found.",
        )

    chunks = _load_chunks_for_doc(document_id)
    if not chunks:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"No chunk data available to build graph for document '{document_id}'.",
        )

    if async_mode:
        background_tasks.add_task(build_graph_for_document, document_id, chunks)
        return {
            "document_id": document_id,
            "status": "processing",
            "message": "Knowledge graph building scheduled in background.",
        }

    try:
        summary = build_graph_for_document(document_id, chunks)
        return {
            "document_id": document_id,
            "status": "ready",
            "summary": summary,
        }
    except Exception as e:
        logger.error("Failed to build graph for doc %s: %s", document_id, e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error building knowledge graph: {str(e)}",
        )


@router.get("/{document_id}/graph")
def get_graph_summary(document_id: str):
    """
    Returns entity, relationship, and graph structure statistics for document_id.
    """
    doc = get_document(document_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{document_id}' not found.",
        )

    store = get_graph_store(document_id)
    stats = store.get_graph_stats()
    return {
        "document_id": document_id,
        "enabled": ENABLE_GRAPHRAG,
        "stats": stats,
    }


@router.get("/{document_id}/graph/query")
def test_graph_query(document_id: str, q: str):
    """
    Executes a test graph traversal query for document_id, returning reasoning paths and entities.
    """
    doc = get_document(document_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{document_id}' not found.",
        )

    res = query_knowledge_graph(document_id, q)
    return {
        "document_id": document_id,
        "question": q,
        "result": res,
    }

