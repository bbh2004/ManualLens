from typing import Any
from backend.knowledge_graph.graph_store import get_graph_store
from backend.knowledge_graph.entity_extractor import extract_entities_and_relations
from backend.utils.logging_config import get_logger

logger = get_logger("manuallens.knowledge_graph.builder")


def build_graph_for_document(
    document_id: str,
    chunks: list[dict[str, Any]],
    max_llm_chunks: int = 40,
) -> dict[str, Any]:
    """
    Extracts entities and relationships from chunks, populates the graph store,
    and connects entities to DocumentPage nodes for citation tracking.
    """
    logger.info("Building Knowledge Graph for document %s...", document_id)

    raw_entities, raw_relationships = extract_entities_and_relations(
        document_id,
        chunks,
        max_llm_chunks=max_llm_chunks,
    )

    store = get_graph_store(document_id)

    # 1. Deduplicate and index entities
    entity_catalog: dict[str, dict[str, Any]] = {}
    for ent in raw_entities:
        name = ent["name"].strip()
        if not name:
            continue
        page = int(ent.get("page_number", 1))
        label = ent.get("label", "Component")
        desc = ent.get("description", "")

        if name not in entity_catalog:
            entity_catalog[name] = {
                "name": name,
                "label": label,
                "descriptions": [desc] if desc else [],
                "page_numbers": {page},
            }
        else:
            entity_catalog[name]["page_numbers"].add(page)
            if desc and desc not in entity_catalog[name]["descriptions"]:
                entity_catalog[name]["descriptions"].append(desc)

    # 2. Add entities to graph store
    for name, data in entity_catalog.items():
        pages = sorted(list(data["page_numbers"]))
        desc = " | ".join(data["descriptions"][:2])
        store.add_node(
            node_id=name,
            label=data["label"],
            properties={
                "name": name,
                "label": data["label"],
                "description": desc,
                "page_numbers": pages,
            },
        )

        # 3. Connect to DocumentPage nodes for citation traceability
        for p in pages:
            page_node_id = f"Page_{p}"
            store.add_node(
                node_id=page_node_id,
                label="DocumentPage",
                properties={
                    "page_number": p,
                    "label": "DocumentPage",
                    "document_id": document_id,
                },
            )
            store.add_edge(
                source_id=name,
                target_id=page_node_id,
                relation="DOCUMENTED_ON",
                properties={"page_number": p, "document_id": document_id},
            )

    # 4. Add relationships
    edges_added = 0
    for rel in raw_relationships:
        src = rel["source"].strip()
        tgt = rel["target"].strip()
        r_type = rel.get("relation", "RELATED_TO").strip()
        p = int(rel.get("page_number", 1))

        if src and tgt and src != tgt:
            store.add_edge(
                source_id=src,
                target_id=tgt,
                relation=r_type,
                properties={"page_number": p, "document_id": document_id},
            )
            edges_added += 1

    # Save to disk
    store.save()
    stats = store.get_graph_stats()

    logger.info(
        "Knowledge Graph build complete for doc %s: %d nodes, %d edges",
        document_id,
        stats.get("nodes_count", 0),
        stats.get("edges_count", 0),
    )

    return {
        "document_id": document_id,
        "nodes_count": stats.get("nodes_count", 0),
        "edges_count": stats.get("edges_count", 0),
        "stats": stats,
    }

