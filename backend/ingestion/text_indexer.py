import pickle
from typing import Any, Optional
import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from backend.config import CHROMA_DIR, BM25_DIR
from backend.utils.logging_config import get_logger

logger = get_logger("manuallens.text_indexer")

_model: Optional[SentenceTransformer] = None
_chroma_client: Optional[chromadb.ClientAPI] = None

def get_embedding_model() -> SentenceTransformer:
    global _model
    if _model is None:
        logger.info("Loading SentenceTransformer model: all-MiniLM-L6-v2")
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model

def get_chroma_client() -> chromadb.ClientAPI:
    global _chroma_client
    if _chroma_client is None:
        logger.info("Initializing ChromaDB PersistentClient at %s", CHROMA_DIR)
        _chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return _chroma_client

def build_text_index(document_id: str, chunks: list[dict[str, Any]]) -> None:
    """
    Generates text embeddings using all-MiniLM-L6-v2 and indexes them in ChromaDB.
    Collection name: doc_{document_id}
    """
    if not chunks:
        logger.warning("No chunks to index for document %s", document_id)
        return

    model = get_embedding_model()
    client = get_chroma_client()
    collection_name = f"doc_{document_id}"

    # Get or create collection
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )

    texts = [c["text"] for c in chunks]
    logger.info("Encoding %d chunks with sentence-transformers...", len(texts))
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=False).tolist()

    ids = [c["id"] for c in chunks]
    metadatas = [
        {
            "document_id": str(c["document_id"]),
            "page_number": int(c["page_number"]),
            "chunk_index": int(c["chunk_index"]),
            "source_type": str(c.get("source_type", "text")),
            "total_pages": int(c.get("total_pages", 0)),
        }
        for c in chunks
    ]

    # Add in batches of 100
    batch_size = 100
    total = len(chunks)
    for i in range(0, total, batch_size):
        end = min(i + batch_size, total)
        collection.upsert(
            ids=ids[i:end],
            documents=texts[i:end],
            embeddings=embeddings[i:end],
            metadatas=metadatas[i:end],
        )

    logger.info(
        "Successfully stored %d chunk vectors in Chroma collection '%s'",
        total,
        collection_name,
    )

def build_bm25_index(document_id: str, chunks: list[dict[str, Any]]) -> None:
    """
    Tokenizes chunk texts, builds BM25Okapi index, and serializes it to disk.
    Saved to storage/db/bm25/{document_id}.pkl
    """
    if not chunks:
        logger.warning("No chunks provided for BM25 index on document %s", document_id)
        return

    logger.info("Building BM25 index for document %s (%d chunks)", document_id, len(chunks))
    # Standard lowercase whitespace + punctuation splitting
    tokenized_corpus = [
        c["text"].lower().split()
        for c in chunks
    ]

    bm25 = BM25Okapi(tokenized_corpus)
    index_data = {
        "bm25": bm25,
        "chunks": chunks,
    }

    bm25_path = BM25_DIR / f"{document_id}.pkl"
    with open(bm25_path, "wb") as f:
        pickle.dump(index_data, f)

    logger.info("Saved BM25 index to %s", bm25_path)

def delete_text_index(document_id: str) -> None:
    """Removes the ChromaDB collection and BM25 pickle for document_id."""
    client = get_chroma_client()
    collection_name = f"doc_{document_id}"
    try:
        client.delete_collection(name=collection_name)
        logger.info("Deleted ChromaDB collection '%s'", collection_name)
    except Exception as e:
        logger.debug("Could not delete Chroma collection '%s': %s", collection_name, e)

    bm25_path = BM25_DIR / f"{document_id}.pkl"
    if bm25_path.exists():
        try:
            bm25_path.unlink()
            logger.info("Deleted BM25 index file %s", bm25_path)
        except Exception as e:
            logger.warning("Could not delete BM25 file %s: %s", bm25_path, e)

