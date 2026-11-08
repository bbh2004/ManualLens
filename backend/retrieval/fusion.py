from typing import Any
from backend.config import RRF_K
from backend.utils.logging_config import get_logger

logger = get_logger("manuallens.retrieval.fusion")

def reciprocal_rank_fusion(
    ranked_lists: list[list[dict[str, Any]]],
    k: int = RRF_K,
) -> list[dict[str, Any]]:
    """
    Combines multiple ranked lists using Reciprocal Rank Fusion (RRF).
    Formula: score(d) = sum(1 / (k + rank_i + 1))
    Returns merged items sorted by RRF score descending, deduplicated by item 'id'.
    """
    scores: dict[str, float] = {}
    chunk_data: dict[str, dict[str, Any]] = {}
    source_attribution: dict[str, set[str]] = {}

    for r_list in ranked_lists:
        for rank, item in enumerate(r_list):
            chunk_id = item["id"]
            reciprocal_rank = 1.0 / (k + rank + 1)
            scores[chunk_id] = scores.get(chunk_id, 0.0) + reciprocal_rank

            if chunk_id not in chunk_data:
                chunk_data[chunk_id] = item
                source_attribution[chunk_id] = {item.get("source", "unknown")}
            else:
                source_attribution[chunk_id].add(item.get("source", "unknown"))

    sorted_ids = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)

    results: list[dict[str, Any]] = []
    for cid in sorted_ids:
        merged_item = dict(chunk_data[cid])
        merged_item["rrf_score"] = round(scores[cid], 6)
        # Indicate if chunk was found by both or single source
        sources = source_attribution[cid]
        if "semantic" in sources and "keyword" in sources:
            merged_item["source"] = "hybrid"
        else:
            merged_item["source"] = next(iter(sources))
        results.append(merged_item)

    logger.debug("RRF fused %d unique chunks from %d lists", len(results), len(ranked_lists))
    return results

