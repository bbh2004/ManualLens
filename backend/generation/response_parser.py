import json
import re
from typing import Any
from backend.utils.logging_config import get_logger

logger = get_logger("manuallens.response_parser")

def parse_response(raw_text: str, evidence: dict[str, Any]) -> dict[str, Any]:
    """
    Parses LLM JSON response, validates citations against evidence, and formats the output
    to match the API specification in PRD Section 4.
    """
    document_id = evidence.get("document_id", "")
    text_chunks = evidence.get("text_chunks", [])
    visual_pages = evidence.get("visual_pages", [])

    # Clean markdown fences if any
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw_text.strip(), flags=re.MULTILINE)
    cleaned = re.sub(r"\s*```$", "", cleaned.strip(), flags=re.MULTILINE)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.warning("Failed to decode LLM response as JSON: %s. Raw: %s", e, raw_text[:200])
        return {
            "answer": raw_text.strip(),
            "citations": [],
            "relevant_pages": [],
            "uncertainties": ["Response could not be parsed as structured JSON."],
        }

    answer = data.get("answer", "")
    uncertainties = data.get("uncertainties", [])
    if isinstance(uncertainties, str):
        uncertainties = [uncertainties]

    # Map retrieved chunks by page number
    page_to_chunk: dict[int, dict[str, Any]] = {}
    for c in text_chunks:
        p_num = c.get("page_number")
        if p_num is not None and p_num not in page_to_chunk:
            page_to_chunk[p_num] = c

    # Parse and validate citations
    raw_citations = data.get("citations", [])
    valid_citations: list[dict[str, Any]] = []

    for c in raw_citations:
        if not isinstance(c, dict):
            continue
        p_num = c.get("page_number")
        if p_num in page_to_chunk:
            matched_chunk = page_to_chunk[p_num]
            valid_citations.append({
                "document_id": document_id,
                "page_number": p_num,
                "chunk_text": c.get("excerpt") or matched_chunk.get("text", "")[:200],
                "source": matched_chunk.get("source", "semantic"),
            })

    # If no citations were parsed by LLM but chunks were retrieved, associate top chunk
    if not valid_citations and text_chunks:
        top_chunk = text_chunks[0]
        valid_citations.append({
            "document_id": document_id,
            "page_number": top_chunk.get("page_number", 1),
            "chunk_text": top_chunk.get("text", "")[:200],
            "source": top_chunk.get("source", "semantic"),
        })

    # Build relevant_pages array
    formatted_pages: list[dict[str, Any]] = []
    seen_pages: set[int] = set()

    for vp in visual_pages:
        p_num = vp.get("page_number")
        if p_num is not None and p_num not in seen_pages:
            seen_pages.add(p_num)
            formatted_pages.append({
                "document_id": document_id,
                "page_number": p_num,
                "image_url": f"/api/pages/{document_id}/{p_num}",
                "visual_score": round(float(vp.get("visual_score", 0.0)), 4),
            })

    # Also include pages referenced in LLM "relevant_pages" if not already present
    llm_pages = data.get("relevant_pages", [])
    if isinstance(llm_pages, list):
        for p in llm_pages:
            if isinstance(p, int) and p not in seen_pages:
                seen_pages.add(p)
                formatted_pages.append({
                    "document_id": document_id,
                    "page_number": p,
                    "image_url": f"/api/pages/{document_id}/{p}",
                    "visual_score": 0.5,
                })

    return {
        "answer": answer,
        "citations": valid_citations,
        "relevant_pages": formatted_pages,
        "uncertainties": uncertainties,
    }

