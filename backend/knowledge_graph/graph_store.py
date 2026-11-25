import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional
import networkx as nx

from backend.config import (
    GRAPH_DIR,
    GRAPH_STORE_TYPE,
    NEO4J_URI,
    NEO4J_USER,
    NEO4J_PASSWORD,
)
from backend.utils.logging_config import get_logger

logger = get_logger("manuallens.knowledge_graph.store")


class GraphStore(ABC):
    """Abstract Base Class for Knowledge Graph storage providers."""

    @abstractmethod
    def add_node(self, node_id: str, label: str, properties: dict[str, Any]) -> None:
        """Adds or updates an entity node in the graph."""
        pass

    @abstractmethod
    def add_edge(
        self,
        source_id: str,
        target_id: str,
        relation: str,
        properties: dict[str, Any],
    ) -> None:
        """Adds a directed relation edge between source and target entities."""
        pass

    @abstractmethod
    def get_node(self, node_id: str) -> Optional[dict[str, Any]]:
        """Retrieves an entity node and its properties by ID."""
        pass

    @abstractmethod
    def get_all_nodes(self) -> list[dict[str, Any]]:
        """Retrieves all entity nodes in the graph."""
        pass

    @abstractmethod
    def multi_hop_query(
        self,
        entity_ids: list[str],
        max_hops: int = 2,
    ) -> list[dict[str, Any]]:
        """
        Traverses relationships starting from entity_ids up to max_hops.
        Returns paths and connected entities with relationship metadata.
        """
        pass

    @abstractmethod
    def get_graph_stats(self) -> dict[str, Any]:
        """Returns entity and relationship counts and distribution statistics."""
        pass

    @abstractmethod
    def delete_graph(self) -> None:
        """Purges all nodes and edges associated with this document."""
        pass

    @abstractmethod
    def save(self) -> None:
        """Persists graph data if using a disk-backed store."""
        pass


class NetworkXGraphStore(GraphStore):
    """
    Local high-performance graph store implemented with NetworkX.
    Persists data as structured JSON in storage/db/graph/{document_id}.json.
    Provides complete zero-dependency, serverless local graph capability.
    """

    def __init__(self, document_id: str):
        self.document_id = document_id
        self.graph = nx.MultiDiGraph()
        self.file_path = GRAPH_DIR / f"{document_id}.json"
        self._load()

    def _load(self) -> None:
        if self.file_path.exists():
            try:
                with open(self.file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.graph = nx.node_link_graph(data, directed=True, multigraph=True)
                logger.debug(
                    "Loaded local graph for %s (%d nodes, %d edges)",
                    self.document_id,
                    self.graph.number_of_nodes(),
                    self.graph.number_of_edges(),
                )
            except Exception as e:
                logger.warning("Failed to load local graph file %s: %s", self.file_path, e)
                self.graph = nx.MultiDiGraph()

    def save(self) -> None:
        try:
            GRAPH_DIR.mkdir(parents=True, exist_ok=True)
            data = nx.node_link_data(self.graph)
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            logger.debug(
                "Persisted local graph for %s to %s (%d nodes, %d edges)",
                self.document_id,
                self.file_path,
                self.graph.number_of_nodes(),
                self.graph.number_of_edges(),
            )
        except Exception as e:
            logger.error("Failed to save local graph for %s: %s", self.document_id, e)

    def add_node(self, node_id: str, label: str, properties: dict[str, Any]) -> None:
        clean_id = node_id.strip()
        if not clean_id:
            return
        props = dict(properties)
        props["id"] = clean_id
        props["label"] = label
        props["document_id"] = self.document_id

        # Merge page numbers if node already exists
        if clean_id in self.graph:
            existing = self.graph.nodes[clean_id]
            existing_pages = set(existing.get("page_numbers", []))
            new_pages = set(props.get("page_numbers", []))
            combined_pages = sorted(list(existing_pages | new_pages))
            props["page_numbers"] = combined_pages

        self.graph.add_node(clean_id, **props)

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        relation: str,
        properties: dict[str, Any],
    ) -> None:
        src = source_id.strip()
        tgt = target_id.strip()
        if not src or not tgt or src == tgt:
            return

        # Ensure endpoints exist
        if src not in self.graph:
            self.graph.add_node(src, id=src, label="Entity", document_id=self.document_id, page_numbers=[])
        if tgt not in self.graph:
            self.graph.add_node(tgt, id=tgt, label="Entity", document_id=self.document_id, page_numbers=[])

        props = dict(properties)
        props["relation"] = relation
        props["document_id"] = self.document_id

        self.graph.add_edge(src, tgt, key=relation, **props)

    def get_node(self, node_id: str) -> Optional[dict[str, Any]]:
        clean_id = node_id.strip()
        if clean_id in self.graph:
            data = dict(self.graph.nodes[clean_id])
            data["id"] = clean_id
            return data
        return None

    def get_all_nodes(self) -> list[dict[str, Any]]:
        nodes = []
        for n, data in self.graph.nodes(data=True):
            item = dict(data)
            item["id"] = n
            nodes.append(item)
        return nodes

    def multi_hop_query(
        self,
        entity_ids: list[str],
        max_hops: int = 2,
    ) -> list[dict[str, Any]]:
        paths_found: list[dict[str, Any]] = []
        seen_edges: set[tuple[str, str, str]] = set()

        # Clean search seeds that exist in graph
        seeds = [eid.strip() for eid in entity_ids if eid.strip() in self.graph]
        if not seeds:
            return []

        for start_node in seeds:
            visited_nodes: set[str] = {start_node}
            frontier: list[tuple[str, list[tuple[str, str, str, dict[str, Any]]]]] = [
                (start_node, [])
            ]

            for _ in range(max_hops):
                next_frontier = []
                for current_node, current_path in frontier:
                    # Outgoing edges
                    for _, neighbor, key, edge_data in self.graph.out_edges(current_node, keys=True, data=True):
                        rel = edge_data.get("relation", key)
                        edge_key = (current_node, neighbor, rel)
                        if edge_key not in seen_edges:
                            seen_edges.add(edge_key)
                            step = (current_node, neighbor, rel, dict(edge_data))
                            new_path = current_path + [step]
                            paths_found.append({
                                "start": start_node,
                                "current": neighbor,
                                "path_length": len(new_path),
                                "steps": new_path,
                            })
                            if neighbor not in visited_nodes and len(new_path) < max_hops:
                                visited_nodes.add(neighbor)
                                next_frontier.append((neighbor, new_path))

                    # Incoming edges (for bidirectional relational context)
                    for predecessor, _, key, edge_data in self.graph.in_edges(current_node, keys=True, data=True):
                        rel = edge_data.get("relation", key)
                        edge_key = (predecessor, current_node, rel)
                        if edge_key not in seen_edges:
                            seen_edges.add(edge_key)
                            step = (predecessor, current_node, rel, dict(edge_data))
                            new_path = current_path + [step]
                            paths_found.append({
                                "start": start_node,
                                "current": predecessor,
                                "path_length": len(new_path),
                                "steps": new_path,
                            })
                            if predecessor not in visited_nodes and len(new_path) < max_hops:
                                visited_nodes.add(predecessor)
                                next_frontier.append((predecessor, new_path))

                frontier = next_frontier

        return paths_found

    def get_graph_stats(self) -> dict[str, Any]:
        node_types: dict[str, int] = {}
        for _, data in self.graph.nodes(data=True):
            lbl = data.get("label", "Unknown")
            node_types[lbl] = node_types.get(lbl, 0) + 1

        rel_types: dict[str, int] = {}
        for _, _, data in self.graph.edges(data=True):
            rel = data.get("relation", "RELATED_TO")
            rel_types[rel] = rel_types.get(rel, 0) + 1

        return {
            "document_id": self.document_id,
            "provider": "networkx_local",
            "nodes_count": self.graph.number_of_nodes(),
            "edges_count": self.graph.number_of_edges(),
            "entity_types": node_types,
            "relationship_types": rel_types,
        }

    def delete_graph(self) -> None:
        self.graph.clear()
        if self.file_path.exists():
            try:
                self.file_path.unlink()
                logger.info("Deleted local graph file %s", self.file_path)
            except Exception as e:
                logger.warning("Failed to delete local graph file %s: %s", self.file_path, e)


class Neo4jGraphStore(GraphStore):
    """
    Neo4j Graph Database provider for cloud or dedicated server deployments.
    Communicates via official neo4j bolt driver and Cypher queries.
    """

    def __init__(self, document_id: str, uri: str, user: str, password: str):
        self.document_id = document_id
        self.uri = uri
        self.user = user
        self.password = password
        self._driver = None
        self._init_driver()

    def _init_driver(self) -> None:
        try:
            from neo4j import GraphDatabase
            self._driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            with self._driver.session() as session:
                session.run("RETURN 1")
            logger.info("Connected to Neo4j database at %s", self.uri)
        except Exception as e:
            logger.error("Failed to connect to Neo4j at %s: %s", self.uri, e)
            self._driver = None
            raise

    def add_node(self, node_id: str, label: str, properties: dict[str, Any]) -> None:
        if not self._driver:
            return
        query = (
            f"MERGE (n:`{label}` {{id: $id, document_id: $doc_id}}) "
            f"SET n += $props"
        )
        with self._driver.session() as session:
            session.run(query, id=node_id, doc_id=self.document_id, props=properties)

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        relation: str,
        properties: dict[str, Any],
    ) -> None:
        if not self._driver:
            return
        query = (
            f"MATCH (a {{id: $src, document_id: $doc_id}}), (b {{id: $tgt, document_id: $doc_id}}) "
            f"MERGE (a)-[r:`{relation}` {{document_id: $doc_id}}]->(b) "
            f"SET r += $props"
        )
        with self._driver.session() as session:
            session.run(
                query,
                src=source_id,
                tgt=target_id,
                doc_id=self.document_id,
                props=properties,
            )

    def get_node(self, node_id: str) -> Optional[dict[str, Any]]:
        if not self._driver:
            return None
        query = "MATCH (n {id: $id, document_id: $doc_id}) RETURN n LIMIT 1"
        with self._driver.session() as session:
            result = session.run(query, id=node_id, doc_id=self.document_id)
            record = result.single()
            if record:
                node = record["n"]
                data = dict(node)
                data["id"] = node_id
                return data
        return None

    def get_all_nodes(self) -> list[dict[str, Any]]:
        if not self._driver:
            return []
        query = "MATCH (n {document_id: $doc_id}) RETURN n"
        nodes = []
        with self._driver.session() as session:
            result = session.run(query, doc_id=self.document_id)
            for record in result:
                node = record["n"]
                data = dict(node)
                nodes.append(data)
        return nodes

    def multi_hop_query(
        self,
        entity_ids: list[str],
        max_hops: int = 2,
    ) -> list[dict[str, Any]]:
        if not self._driver or not entity_ids:
            return []
        query = (
            f"MATCH path = (start {{document_id: $doc_id}})-[*1..{max_hops}]-(connected {{document_id: $doc_id}}) "
            f"WHERE start.id IN $ids "
            f"RETURN path LIMIT 50"
        )
        paths_found = []
        with self._driver.session() as session:
            result = session.run(query, doc_id=self.document_id, ids=entity_ids)
            for record in result:
                p = record["path"]
                steps = []
                for rel in p.relationships:
                    steps.append((
                        rel.start_node["id"],
                        rel.end_node["id"],
                        rel.type,
                        dict(rel),
                    ))
                paths_found.append({
                    "start": p.start_node["id"],
                    "current": p.end_node["id"],
                    "path_length": len(steps),
                    "steps": steps,
                })
        return paths_found

    def get_graph_stats(self) -> dict[str, Any]:
        if not self._driver:
            return {"provider": "neo4j", "error": "disconnected"}
        with self._driver.session() as session:
            n_res = session.run("MATCH (n {document_id: $doc_id}) RETURN count(n) as cnt", doc_id=self.document_id)
            nodes_cnt = n_res.single()["cnt"]
            e_res = session.run("MATCH ()-[r {document_id: $doc_id}]->() RETURN count(r) as cnt", doc_id=self.document_id)
            edges_cnt = e_res.single()["cnt"]

        return {
            "document_id": self.document_id,
            "provider": "neo4j",
            "nodes_count": nodes_cnt,
            "edges_count": edges_cnt,
        }

    def delete_graph(self) -> None:
        if not self._driver:
            return
        query = "MATCH (n {document_id: $doc_id}) DETACH DELETE n"
        with self._driver.session() as session:
            session.run(query, doc_id=self.document_id)
        logger.info("Deleted Neo4j graph for document %s", self.document_id)

    def save(self) -> None:
        pass


_store_instances: dict[str, GraphStore] = {}


def get_graph_store(document_id: str) -> GraphStore:
    """
    Factory function providing plug-and-play access to the configured graph store.
    Automatically instantiates Neo4j if configured and reachable; otherwise defaults
    to the reliable local NetworkX persistent file store.
    """
    if document_id in _store_instances:
        return _store_instances[document_id]

    store_type = GRAPH_STORE_TYPE.lower()

    if store_type in ("neo4j", "auto") and NEO4J_URI:
        try:
            store = Neo4jGraphStore(
                document_id=document_id,
                uri=NEO4J_URI,
                user=NEO4J_USER,
                password=NEO4J_PASSWORD,
            )
            _store_instances[document_id] = store
            return store
        except Exception as e:
            logger.warning(
                "Could not connect to Neo4j (%s). Falling back to local NetworkX graph store.",
                e,
            )

    # Local file-based NetworkX store
    store = NetworkXGraphStore(document_id=document_id)
    _store_instances[document_id] = store
    return store


def delete_graph_store(document_id: str) -> None:
    """Purges graph store instance and underlying data for document_id."""
    store = _store_instances.pop(document_id, None)
    if store:
        store.delete_graph()
    else:
        # Fallback disk cleanup
        f_path = GRAPH_DIR / f"{document_id}.json"
        if f_path.exists():
            f_path.unlink()

