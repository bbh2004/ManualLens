import time
from typing import Any, Optional
from groq import Groq

from backend.config import GROQ_API_KEY, GROQ_MODEL, LLM_MAX_TOKENS, LLM_TEMPERATURE
from backend.utils.logging_config import get_logger

logger = get_logger("manuallens.generation.groq")

_client: Optional[Groq] = None

# Fallback models if primary model is unavailable or rate-limited
MODEL_FALLBACKS = [
    GROQ_MODEL,
    "qwen/qwen3.8-27b",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "llama-3.1-8b-instant",
]

def get_client() -> Groq:
    """Returns singleton Groq API client."""
    global _client
    if _client is None:
        if not GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY is not set. Please provide a valid Groq API key in .env"
            )
        _client = Groq(api_key=GROQ_API_KEY)
    return _client

def call_groq(messages: list[dict[str, str]], model_override: Optional[str] = None) -> str:
    """
    Calls Groq chat completion API with retry on rate limit and automatic model fallback.
    """
    client = get_client()
    target_models = [model_override] if model_override else MODEL_FALLBACKS

    # Remove duplicates preserving order
    models_to_try: list[str] = []
    for m in target_models:
        if m and m not in models_to_try:
            models_to_try.append(m)

    last_error: Optional[Exception] = None

    for model in models_to_try:
        for attempt in range(3):
            try:
                logger.debug("Calling Groq API (model=%s, attempt=%d)", model, attempt + 1)
                completion = client.chat.completions.create(
                    model=model,
                    messages=messages,  # type: ignore
                    max_tokens=LLM_MAX_TOKENS,
                    temperature=LLM_TEMPERATURE,
                    response_format={"type": "json_object"},
                )
                content = completion.choices[0].message.content or ""
                return content
            except Exception as e:
                err_str = str(e).lower()
                last_error = e
                if "rate_limit" in err_str or "429" in err_str:
                    logger.warning("Groq rate limit encountered. Waiting before retry...")
                    time.sleep(2 * (attempt + 1))
                elif "model" in err_str and ("not found" in err_str or "decommissioned" in err_str):
                    logger.warning("Model '%s' unavailable. Trying fallback model...", model)
                    break  # Break attempt loop, try next model
                else:
                    logger.error("Groq API error on model %s: %s", model, e)
                    if attempt == 2:
                        break
                    time.sleep(1)

    raise RuntimeError(f"All Groq API attempts failed. Last error: {last_error}")
