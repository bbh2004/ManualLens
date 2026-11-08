from typing import Any
from backend.utils.logging_config import get_logger

logger = get_logger("manuallens.prompt_builder")

SYSTEM_PROMPT = """You are ManualLens, an expert technical manual assistant.
You answer user questions strictly based on the retrieved manual excerpts and page context provided below.

Rules:
1. Grounding: Only state facts, procedures, part numbers, and specifications explicitly present in the provided excerpts.
2. Unanswerable queries: If the manual excerpts do not contain the answer or only partially answer it, clearly state what is missing and list specific gaps in the "uncertainties" array. Do NOT invent or assume details.
3. Citations: Every factual claim in the "answer" MUST cite its source page in square brackets, e.g. [Page 12].
4. Procedures: When asked for procedures or troubleshooting steps, present them in clear, sequential ordered steps.
5. Exact Identifiers: Maintain exact case and punctuation for error codes, part numbers, and model identifiers.
6. Return format: Output valid JSON adhering exactly to this schema:
{
  "answer": "Grounded answer text with inline [Page X] citations.",
  "citations": [
    {
      "page_number": 12,
      "excerpt": "Exact or near-exact short quotation from the text"
    }
  ],
  "relevant_pages": [12, 14],
  "uncertainties": ["Statement of what the manual does not cover, or empty array if fully covered."]
}"""

def build_prompt(evidence: dict[str, Any]) -> list[dict[str, str]]:
    """
    Assembles evidence (question, text_chunks, visual_pages) into Groq chat messages.
    """
    question = evidence.get("question", "")
    text_chunks = evidence.get("text_chunks", [])
    visual_pages = evidence.get("visual_pages", [])

    context_parts: list[str] = []

    context_parts.append(f"### USER QUESTION:\n{question}\n")

    if text_chunks:
        context_parts.append("### RETRIEVED MANUAL EXCERPTS:")
        for idx, chunk in enumerate(text_chunks, 1):
            page_num = chunk.get("page_number", "?")
            src = chunk.get("source", "text")
            text = chunk.get("text", "").strip()
            context_parts.append(
                f"[Excerpt {idx} | Page {page_num} | Source: {src}]\n{text}\n"
            )
    else:
        context_parts.append("### RETRIEVED MANUAL EXCERPTS:\n(No relevant text excerpts found)\n")

    if visual_pages:
        context_parts.append("### RELEVANT VISUAL PAGES (Diagrams / Schematics / Tables):")
        for vp in visual_pages:
            p_num = vp.get("page_number")
            v_score = vp.get("visual_score", 0.0)
            context_parts.append(f"- Page {p_num} (Visual relevance score: {v_score})")

    context_parts.append(
        "\nProvide your grounded response in the specified JSON format."
    )

    user_content = "\n".join(context_parts)

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

