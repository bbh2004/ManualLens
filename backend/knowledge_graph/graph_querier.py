import re
from typing import Any
from backend.config import GRAPH_TOP_K
from backend.knowledge_graph.graph_store import get_graph_store
from backend.utils.logging_config import get_logger

logger = get_logger("manuallens.knowledge_graph.querier")


def query_knowledge_graph(
    document_id: str,
    question: str,
    max_hops: int = 2,
    top_k: int = GRAPH_TOP_K,
) -> dict[str, Any]:
    """
    Identifies relevant entity seeds in the question, executes multi-hop graph traversal,
    and returns connected reasoning paths, linked page numbers, and structured evidence.
    """
    store = get_graph_store(document_id)
    all_nodes = store.get_all_nodes()

    if not all_nodes:
        logger.debug("No graph nodes available for document %s", document_id)
        return {
            "matched_entities": [],
            "paths": [],
            "page_numbers": [],
            "graph_evidence": "",
            "citations": [],
        }

    q_lower = question.lower()
    q_tokens = set(re.findall(r"\b\w+\b", q_lower))

    # 1. Match entities from question against graph nodes
    matched_seeds: list[dict[str, Any]] = []

    for node in all_nodes:
        node_id = node.get("id", "")
        node_label = node.get("label", "")
        n_lower = node_id.lower()

        # Skip raw Page_X nodes as seeds
        if node_label == "DocumentPage" or node_id.startswith("Page_"):
            continue

        # Exact substring match (e.g. "e50", "atmega2560", "amber led", "fuser")
        if n_lower in q_lower or (len(n_lower) > 3 and any(n_lower in tok or tok in n_lower for tok in q_tokens)):
            matched_seeds.append(node)
        elif node_id in question:
            matched_seeds.append(node)

    logger.debug("Found %d matching graph entity seeds for question", len(matched_seeds))

    if not matched_seeds:
        # Fallback: check prominent error code / component regex tokens in question
        code_match = re.search(r"\b([eE]\d{1,3}|[A-Z]{2,6}\s+LED)\b", question)
        if code_match:
            term = code_match.group(1).lower()
            for node in all_nodes:
                if term in node.get("id", "").lower():
                    matched_seeds.append(node)

    seed_ids = [s["id"] for s in matched_seeds[:5]]

    if not seed_ids:
        return {
            "matched_entities": [],
            "paths": [],
            "page_numbers": [],
            "graph_evidence": "",
            "citations": [],
        }

    # 2. Execute multi-hop traversal from seeds
    paths = store.multi_hop_query(entity_ids=seed_ids, max_hops=max_hops)

    # 3. Format reasoning chains and collect page citations
    formatted_paths: list[str] = []
    associated_pages: set[int] = set()
    citations: list[dict[str, Any]] = []

    # Include seed node properties
    for s in matched_seeds[:top_k]:
        for p in s.get("page_numbers", []):
            try:
                associated_pages.add(int(p))
            except (ValueError, TypeError):
                pass

    for p in paths[:top_k * 2]:
        steps = p.get("steps", [])
        if not steps:
            continue

        chain_segments = []
        for src, tgt, rel, edge_props in steps:
            chain_segments.append(f"({src}) -[{rel}]-> ({tgt})")
            p_num = edge_props.get("page_number")
            if p_num is not None:
                try:
                    associated_pages.add(int(p_num))
                except (ValueError, TypeError):
                    pass
            if tgt.startswith("Page_"):
                try:
                    associated_pages.add(int(tgt.replace("Page_", "")))
                except (ValueError, TypeError):
                    pass

        path_str = " => ".join(chain_segments)
        if path_str not in formatted_paths:
            formatted_paths.append(path_str)

    # Build concise textual graph evidence for LLM prompt
    evidence_lines: list[str] = []
    for s in matched_seeds[:top_k]:
        s_id = s["id"]
        s_lbl = s.get("label", "Entity")
        s_desc = s.get("description", "")
        s_pages = s.get("page_numbers", [])
        evidence_lines.append(
            f"• Entity: [{s_lbl}] {s_id} (Cited Pages: {s_pages})"
            + (f" — {s_desc}" if s_desc else "")
        )

    if formatted_paths:
        evidence_lines.append("\n  Multi-Hop Reasoning Paths:")
        for fp in formatted_paths[:top_k]:
            evidence_lines.append(f"  - {fp}")

    graph_evidence_text = "\n".join(evidence_lines)

    # Create citations for pages confirmed by graph relationships
    for page_num in sorted(list(associated_pages))[:5]:
        citations.append({
            "document_id": document_id,
            "page_number": page_num,
            "chunk_text": f"Knowledge graph entity path confirmed on Page {page_num}",
            "source": "graph",
        })

    logger.debug(
        "Graph traversal for doc %s returned %d paths, linked to pages %s",
        document_id,
        len(formatted_paths),
        sorted(list(associated_pages)),
    )

    return {
        "matched_entities": [s["id"] for s in matched_seeds],
        "paths": formatted_paths[:top_k],
        "page_numbers": sorted(list(associated_pages)),
        "graph_evidence": graph_evidence_text,
        "citations": citations,
    }

