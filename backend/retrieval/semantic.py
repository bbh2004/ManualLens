from typing import Any
from backend.ingestion.text_indexer import get_embedding_model, get_chroma_client
from backend.config import TEXT_TOP_K
from backend.utils.logging_config import get_logger

logger = get_logger("manuallens.retrieval.semantic")

def semantic_search(
    document_id: str,
    question: str,
    top_k: int = TEXT_TOP_K,
) -> list[dict[str, Any]]:
    """
    Performs vector similarity search against ChromaDB for document_id.
    Returns ranked list of top_k chunks.
    """
    logger.debug("Executing semantic search for document %s: '%s'", document_id, question)
    client = get_chroma_client()
    collection_name = f"doc_{document_id}"

    try:
        collection = client.get_collection(name=collection_name)
    except Exception as e:
        logger.error("Chroma collection '%s' not found: %s", collection_name, e)
        return []

    model = get_embedding_model()
    query_vector = model.encode(question).tolist()

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]
    ids = results.get("ids", [[]])[0]

    output: list[dict[str, Any]] = []
    for chunk_id, text, meta, dist in zip(ids, docs, metas, distances):
        # In cosine distance: score = 1 - distance
        score = max(0.0, 1.0 - float(dist)) if dist is not None else 0.0
        output.append({
            "id": chunk_id,
            "text": text,
            "page_number": int(meta.get("page_number", 1)),
            "chunk_index": int(meta.get("chunk_index", 0)),
            "document_id": document_id,
            "score": round(score, 4),
            "source": "semantic",
        })

    logger.debug("Semantic search returned %d candidates", len(output))
    return output

