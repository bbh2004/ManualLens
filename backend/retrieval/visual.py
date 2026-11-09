from pathlib import Path
from typing import Any
from backend.config import VISUAL_TOP_K, USE_MODAL
from backend.ingestion.visual_indexer import get_byaldi_index_root
from backend.utils.logging_config import get_logger

logger = get_logger("manuallens.retrieval.visual")

_index_cache: dict[str, Any] = {}

def visual_search(
    document_id: str,
    question: str,
    top_k: int = VISUAL_TOP_K,
) -> list[dict[str, Any]]:
    """
    Executes late-interaction visual search using ColPali against rendered page images.
    If USE_MODAL is true, delegates search to the serverless GPU worker on Modal.
    Otherwise, runs search using the local byaldi index.
    Returns ranked list of visually relevant pages and their scores.
    """
    if USE_MODAL:
        try:
            from backend.modal_worker import get_modal_worker
            worker = get_modal_worker()
            logger.debug("Executing Modal cloud visual search for doc '%s'...", document_id)
            raw_results = worker.search_pages.remote(document_id, question, top_k)
            output: list[dict[str, Any]] = []
            for r in raw_results:
                page_num = int(r.get("page_number", 1))
                score = float(r.get("visual_score", 0.0))
                output.append({
                    "document_id": document_id,
                    "page_number": page_num,
                    "visual_score": round(score, 4),
                    "image_url": f"/api/pages/{document_id}/{page_num}",
                })
            logger.debug("Modal visual search returned %d candidate pages", len(output))
            return output
        except Exception as e:
            logger.warning("Modal visual search failed for document %s: %s", document_id, e)
            return []

    # Local CPU execution fallback
    index_name = f"vis_{document_id}"
    index_root = get_byaldi_index_root()
    index_path = index_root / index_name
    default_path = Path(".byaldi") / index_name

    active_root = None
    if index_path.exists():
        active_root = str(index_root)
    elif default_path.exists():
        active_root = ".byaldi"
    else:
        logger.debug("No local visual index found for document %s (checked %s and %s)", document_id, index_path, default_path)
        return []

    try:
        import torch
        from byaldi import RAGMultiModalModel

        device = "cuda" if torch.cuda.is_available() else "cpu"
        if index_name not in _index_cache:
            logger.info("Loading local visual index '%s' from %s", index_name, active_root)
            model = RAGMultiModalModel.from_index(
                index_path=index_name,
                index_root=active_root,
                device=device,
                verbose=0,
            )
            _index_cache[index_name] = model
        else:
            model = _index_cache[index_name]

        raw_results = model.search(question, k=top_k)

        if isinstance(raw_results, list) and raw_results and isinstance(raw_results[0], list):
            results_list = raw_results[0]
        else:
            results_list = raw_results

        output = []
        doc_names = getattr(model.model, "doc_ids_to_file_names", {}) or {}

        for r in results_list:
            doc_id_val = getattr(r, "doc_id", None)
            score = getattr(r, "score", 0.0)
            page_num = getattr(r, "page_num", 1)
            if doc_id_val is not None and doc_id_val in doc_names:
                fname = str(doc_names[doc_id_val])
                import re
                m = re.search(r"page_(\d+)\.png", fname)
                if m:
                    page_num = int(m.group(1))

            output.append({
                "document_id": document_id,
                "page_number": int(page_num),
                "visual_score": round(float(score), 4),
                "image_url": f"/api/pages/{document_id}/{page_num}",
            })

        logger.debug("Local visual search returned %d candidate pages", len(output))
        return output

    except Exception as e:
        logger.warning("Local visual search failed for document %s: %s", document_id, e)
        return []
