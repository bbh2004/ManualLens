from typing import Any
import tiktoken
from backend.config import CHUNK_SIZE_TOKENS, CHUNK_OVERLAP_TOKENS
from backend.utils.logging_config import get_logger

logger = get_logger("manuallens.chunker")

def chunk_pages(
    document_id: str,
    pages: list[dict[str, Any]],
    chunk_size: int = CHUNK_SIZE_TOKENS,
    overlap: int = CHUNK_OVERLAP_TOKENS,
) -> list[dict[str, Any]]:
    """
    Splits extracted page texts into token-bounded chunks using tiktoken (cl100k_base).
    Crucial: Chunks do NOT cross page boundaries (ADR-011).
    """
    logger.info(
        "Chunking %d pages for document %s (chunk_size=%d, overlap=%d)",
        len(pages),
        document_id,
        chunk_size,
        overlap,
    )
    encoder = tiktoken.get_encoding("cl100k_base")
    chunks: list[dict[str, Any]] = []

    step = max(1, chunk_size - overlap)

    for page in pages:
        page_num = page["page_number"]
        text = page.get("text", "").strip()
        source_type = page.get("source_type", "text")
        total_pages = page.get("total_pages", len(pages))

        if not text:
            continue

        tokens = encoder.encode(text)
        total_tokens = len(tokens)

        if total_tokens <= chunk_size:
            chunk_id = f"{document_id}_p{page_num}_c0"
            chunks.append({
                "id": chunk_id,
                "document_id": document_id,
                "page_number": page_num,
                "chunk_index": 0,
                "text": text,
                "source_type": source_type,
                "total_pages": total_pages,
                "token_count": total_tokens,
            })
            continue

        chunk_idx = 0
        start = 0
        while start < total_tokens:
            end = min(start + chunk_size, total_tokens)
            window_tokens = tokens[start:end]
            chunk_text = encoder.decode(window_tokens).strip()

            if chunk_text:
                chunk_id = f"{document_id}_p{page_num}_c{chunk_idx}"
                chunks.append({
                    "id": chunk_id,
                    "document_id": document_id,
                    "page_number": page_num,
                    "chunk_index": chunk_idx,
                    "text": chunk_text,
                    "source_type": source_type,
                    "total_pages": total_pages,
                    "token_count": len(window_tokens),
                })
                chunk_idx += 1

            if end >= total_tokens:
                break
            start += step

    logger.info("Generated %d chunks for document %s", len(chunks), document_id)
    return chunks

