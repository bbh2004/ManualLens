import pickle
from typing import Any
from backend.config import BM25_DIR, TEXT_TOP_K
from backend.utils.logging_config import get_logger

logger = get_logger("manuallens.retrieval.keyword")

def keyword_search(
    document_id: str,
    question: str,
    top_k: int = TEXT_TOP_K,
) -> list[dict[str, Any]]:
    """
    Performs BM25 keyword search over indexed chunks for document_id.
    """
    bm25_path = BM25_DIR / f"{document_id}.pkl"
    if not bm25_path.exists():
        logger.warning("BM25 index not found at %s", bm25_path)
        return []

    try:
        with open(bm25_path, "rb") as f:
            data = pickle.load(f)
    except Exception as e:
        logger.error("Failed to load BM25 index from %s: %s", bm25_path, e)
        return []

    bm25 = data["bm25"]
    chunks = data["chunks"]

    tokenized_query = question.lower().split()
    if not tokenized_query:
        return []

    scores = bm25.get_scores(tokenized_query)

    # Pair each chunk with its score
    scored_chunks = list(zip(chunks, scores))
    # Sort descending by score
    scored_chunks.sort(key=lambda x: x[1], reverse=True)

    results: list[dict[str, Any]] = []
    for chunk, score in scored_chunks[:top_k]:
        if score <= 0.0 and len(results) >= 1:
            # Skip zero-score chunks if we already have matching candidates
            continue
        results.append({
            "id": chunk["id"],
            "text": chunk["text"],
            "page_number": int(chunk["page_number"]),
            "chunk_index": int(chunk.get("chunk_index", 0)),
            "document_id": document_id,
            "score": round(float(score), 4),
            "source": "keyword",
        })

    logger.debug("BM25 search returned %d candidates", len(results))
    return results

