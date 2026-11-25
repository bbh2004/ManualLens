import json
import re
from typing import Any, Optional
from backend.generation.groq_client import call_groq
from backend.utils.logging_config import get_logger

logger = get_logger("manuallens.knowledge_graph.extractor")

EXTRACTION_SYSTEM_PROMPT = """You are an expert technical knowledge graph extraction engine for service manuals and engineering datasheets.
Your task is to extract technical entities and directional relationships from the provided manual text.

Valid Entity Labels:
- Device: The overarching system, product, or board (e.g. 'Arduino Mega 2560', 'Catalyst 2960-X', 'MFC-L2710DW')
- Component: Physical parts, chips, connectors, ports, modules (e.g. 'ATmega2560', 'Fuser Unit', 'Drum Unit', 'SYST LED', 'Paper Sensor', 'DC Jack', 'Power Supply')
- ErrorCode: Error messages, status codes, beep codes, or LED patterns (e.g. 'E50', 'E6', 'E17', 'Solid Amber LED', 'Flashing Green SYST LED', 'No Paper')
- Symptom: Observable issue or behavioral state (e.g. 'Overheating', 'Laser diode overheated', 'POST in progress', 'Paper feed failure', 'STP blocked')
- Procedure: Ordered diagnostic, repair, cleaning, or configuration action (e.g. 'Reset Drum Counter', 'Inspect Paper Sensor', 'Replace Fuser Unit', 'Power Off Device')
- Specification: Electrical, mechanical, or thermal parameters (e.g. '16 MHz', '7-12V Input', '100-240V AC', '54 Digital Pins')

Valid Relationship Types:
- HAS_COMPONENT: (Device) -> (Component)
- ASSOCIATED_WITH_ERROR: (Component or Device) -> (ErrorCode)
- HAS_SYMPTOM: (ErrorCode or Component) -> (Symptom)
- DIAGNOSED_BY: (ErrorCode or Symptom) -> (Procedure)
- FIXED_BY: (ErrorCode or Symptom or Component) -> (Procedure)
- HAS_SPECIFICATION: (Device or Component) -> (Specification)

Output strictly valid JSON with this exact schema:
{
  "entities": [
    {
      "name": "Standardized concise entity name",
      "label": "Device|Component|ErrorCode|Symptom|Procedure|Specification",
      "description": "Short 1-sentence technical description"
    }
  ],
  "relationships": [
    {
      "source": "Source entity name",
      "relation": "HAS_COMPONENT|ASSOCIATED_WITH_ERROR|HAS_SYMPTOM|DIAGNOSED_BY|FIXED_BY|HAS_SPECIFICATION",
      "target": "Target entity name"
    }
  ]
}
Rules:
- Be precise and faithful to the text. Do not invent entities.
- Keep entity names concise, specific, and title-cased or canonical (e.g. 'E50 Error', 'Fuser Unit').
- If no meaningful technical entities or relations exist in the excerpt, return empty arrays."""


def _regex_heuristic_extraction(text: str, page_number: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Fast rule-based extractor targeting prominent manual identifiers:
    error codes (E50, E6, E17, E42), LED statuses, microcontrollers, voltages, frequencies.
    Acts as a deterministic safety net.
    """
    entities: list[dict[str, Any]] = []
    relationships: list[dict[str, Any]] = []

    # 1. Error Codes (e.g., Error E50, E6, E42, Code 0x...)
    error_matches = re.finditer(r"\b(?:Error\s+|Code\s+)?([E|e]\d{1,3})\b", text)
    for m in error_matches:
        code_name = f"Error {m.group(1).upper()}"
        entities.append({
            "name": code_name,
            "label": "ErrorCode",
            "page_number": page_number,
            "description": f"Diagnostic error code mentioned on page {page_number}",
        })

    # 2. LED Indicators (e.g., SYST LED, amber LED, STAT LED)
    led_matches = re.finditer(r"\b([A-Z]{2,6}\s+LED|amber\s+LED|green\s+LED)\b", text, re.IGNORECASE)
    for m in led_matches:
        led_name = m.group(1).strip()
        entities.append({
            "name": led_name,
            "label": "Component",
            "page_number": page_number,
            "description": f"Status LED indicator mentioned on page {page_number}",
        })

    # 3. Microcontrollers / Chips (e.g., ATmega2560, ATmega16U2)
    chip_matches = re.finditer(r"\b(ATmega\w+|Cisco\s+Catalyst\s+[\w\-]+|Brother\s+[\w\-]+)\b", text, re.IGNORECASE)
    for m in chip_matches:
        chip_name = m.group(1).strip()
        entities.append({
            "name": chip_name,
            "label": "Component" if "atmega" in chip_name.lower() else "Device",
            "page_number": page_number,
            "description": f"Hardware component/device on page {page_number}",
        })

    # 4. Critical Specifications (e.g., 16 MHz, 100 to 240 VAC, 7-12V)
    spec_matches = re.finditer(r"\b(\d+(?:\.\d+)?\s*(?:MHz|GHz|VAC|VDC|mA|pins?|VAC))\b", text, re.IGNORECASE)
    for m in spec_matches:
        spec_val = m.group(1).strip()
        entities.append({
            "name": spec_val,
            "label": "Specification",
            "page_number": page_number,
            "description": f"Technical parameter on page {page_number}",
        })

    return entities, relationships


def extract_from_chunk(
    chunk_text: str,
    page_number: int,
    use_llm: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Extracts entities and relationships from a single text chunk using Groq LLM
    combined with heuristic rule-based pattern matching.
    """
    entities: list[dict[str, Any]] = []
    relationships: list[dict[str, Any]] = []

    # 1. Apply heuristic regex extraction
    h_entities, h_relations = _regex_heuristic_extraction(chunk_text, page_number)
    entities.extend(h_entities)
    relationships.extend(h_relations)

    if not use_llm or len(chunk_text.strip()) < 40:
        return entities, relationships

    # 2. Apply Groq LLM Extraction
    prompt_messages = [
        {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Extract technical entities and relationships from this manual page (Page {page_number}):\n\n{chunk_text}",
        },
    ]

    try:
        raw_response = call_groq(prompt_messages)
        cleaned = re.sub(r"^```(?:json)?\s*", "", raw_response.strip(), flags=re.MULTILINE)
        cleaned = re.sub(r"\s*```$", "", cleaned.strip(), flags=re.MULTILINE)
        data = json.loads(cleaned)

        raw_entities = data.get("entities", [])
        raw_relations = data.get("relationships", [])

        for ent in raw_entities:
            name = ent.get("name", "").strip()
            label = ent.get("label", "Component").strip()
            desc = ent.get("description", "")
            if name:
                entities.append({
                    "name": name,
                    "label": label,
                    "page_number": page_number,
                    "description": desc,
                })

        for rel in raw_relations:
            src = rel.get("source", "").strip()
            relation = rel.get("relation", "RELATED_TO").strip().upper()
            tgt = rel.get("target", "").strip()
            if src and tgt and src != tgt:
                relationships.append({
                    "source": src,
                    "relation": relation,
                    "target": tgt,
                    "page_number": page_number,
                })

    except Exception as e:
        logger.warning(
            "LLM entity extraction failed on page %d: %s. Relying on heuristic extraction.",
            page_number,
            e,
        )

    return entities, relationships


def extract_entities_and_relations(
    document_id: str,
    chunks: list[dict[str, Any]],
    max_llm_chunks: int = 40,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Extracts entities and relationships across all chunks of a document.
    To balance speed and LLM rate-limits on massive manuals (e.g. 672-page service manual),
    prioritizes informative chunks (diagnostics, troubleshooting, specifications, error codes)
    for LLM extraction while running heuristic extraction over 100% of chunks.
    """
    logger.info("Extracting knowledge graph entities for doc %s (%d chunks)...", document_id, len(chunks))

    all_entities: list[dict[str, Any]] = []
    all_relationships: list[dict[str, Any]] = []

    # Identify chunks containing diagnostic or key technical terms for prioritized LLM processing
    key_patterns = re.compile(
        r"\b(error|code|troubleshoot|diagnos|step|procedure|specification|voltage|fan|fuser|sensor|led|pin|replace|inspect)\b",
        re.IGNORECASE,
    )

    scored_chunks = []
    for c in chunks:
        text = c.get("text", "")
        matches = len(key_patterns.findall(text))
        scored_chunks.append((matches, c))

    # Sort descending by relevance density
    scored_chunks.sort(key=lambda x: x[0], reverse=True)

    llm_processed_count = 0

    for score, chunk in scored_chunks:
        text = chunk.get("text", "")
        page_num = chunk.get("page_number", 1)

        # Decide whether to invoke LLM
        use_llm = (llm_processed_count < max_llm_chunks) and (score > 0)
        if use_llm:
            llm_processed_count += 1

        ents, rels = extract_from_chunk(text, page_num, use_llm=use_llm)
        all_entities.extend(ents)
        all_relationships.extend(rels)

    logger.info(
        "Extracted %d raw entities and %d relationships for doc %s (LLM parsed %d chunks)",
        len(all_entities),
        len(all_relationships),
        document_id,
        llm_processed_count,
    )

    return all_entities, all_relationships

