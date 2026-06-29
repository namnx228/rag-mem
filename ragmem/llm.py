"""Factory for the GraphRAG extraction LLM (Anthropic Haiku via OpenRouter).

Imported lazily so that importing ``ragmem`` does not pull in llama-index unless
GraphRAG is actually used. Routes through OpenRouter using
``llama-index-llms-openai`` with a custom base URL. Reads
``OPENROUTER_API_KEY`` from the environment when no key is passed.
"""

from __future__ import annotations

import os
from typing import Any

DEFAULT_EXTRACTION_MODEL = "anthropic/claude-haiku-4-5"

_DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"


def make_extraction_llm(model: str = DEFAULT_EXTRACTION_MODEL, api_key: str | None = None) -> Any:
    from llama_index.llms.openai import OpenAI as LlamaOpenAI

    resolved_key = api_key or os.environ.get("OPENROUTER_API_KEY")
    return LlamaOpenAI(
        model=model,
        api_base=_DEFAULT_BASE_URL,
        api_key=resolved_key,
    )
