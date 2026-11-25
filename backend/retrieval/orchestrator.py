from typing import Any
from backend.config import TEXT_TOP_K, ENABLE_GRAPHRAG
from backend.retrieval.semantic import semantic_search
from backend.retrieval.keyword import keyword_search
from backend.retrieval.fusion import reciprocal_rank_fusion
from backend.generation.prompt_builder import build_prompt
from backend.generation.groq_client import call_groq
from backend.generation.response_parser import parse_response
from backend.utils.logging_config import get_logger

logger = get_logger("manuallens.retrieval.orchestrator")

def answer_question(document_id: str, question: str) -> dict[str, Any]:
    """
    Main orchestration function for multi-path retrieval and answer generation.
    1. Executes semantic search (ChromaDB)
    2. Executes BM25 keyword search
    3. Fuses text results via RRF
    4. Executes visual search (ColPali)
    5. Executes GraphRAG multi-hop traversal (if ENABLE_GRAPHRAG=True)
    6. Assembles evidence and prompts Groq LLM
    7. Parses, grounds, and formats response
    """
    logger.info("Answering query for doc %s: '%s'", document_id, question)

    # 1. Semantic search
    semantic_results = semantic_search(document_id, question, top_k=TEXT_TOP_K)

    # 2. Keyword search
    keyword_results = keyword_search(document_id, question, top_k=TEXT_TOP_K)

    # 3. Hybrid fusion
    fused_text = reciprocal_rank_fusion([semantic_results, keyword_results], k=60)[:TEXT_TOP_K]

    # 4. Visual search (ColPali)
    visual_results: list[dict[str, Any]] = []
    try:
        from backend.retrieval.visual import visual_search
        visual_results = visual_search(document_id, question)
    except (ImportError, Exception) as e:
        logger.debug("Visual search not available or empty: %s", e)

    # 5. GraphRAG search (Part 2 Feature 1)
    graph_data: dict[str, Any] = {}
    if ENABLE_GRAPHRAG:
        try:
            from backend.knowledge_graph.graph_querier import query_knowledge_graph
            graph_data = query_knowledge_graph(document_id, question)
            logger.debug(
                "GraphRAG retrieved %d paths and %d pages for query",
                len(graph_data.get("paths", [])),
                len(graph_data.get("page_numbers", [])),
            )
        except Exception as e:
            logger.warning("GraphRAG query failed for doc %s: %s", document_id, e)

    # 6. Assemble evidence
    evidence = {
        "document_id": document_id,
        "question": question,
        "text_chunks": fused_text,
        "visual_pages": visual_results,
        "graph_data": graph_data,
    }

    # 7. Build prompt and query Groq
    messages = build_prompt(evidence)
    raw_response = call_groq(messages)

    # 8. Ground and validate response
    response = parse_response(raw_response, evidence)
    return response

