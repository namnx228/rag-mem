"""Factory for the GraphRAG extraction LLM (Anthropic Haiku, via llama-index).

Imported lazily so that importing ``ragmem`` does not pull in llama-index unless
GraphRAG is actually used. Reads ``ANTHROPIC_API_KEY`` from the environment when
no key is passed.
"""

from __future__ import annotations

from typing import Any

DEFAULT_EXTRACTION_MODEL = "claude-haiku-4-5"


def make_extraction_llm(model: str = DEFAULT_EXTRACTION_MODEL, api_key: str | None = None) -> Any:
    from llama_index.llms.anthropic import Anthropic

    kwargs: dict[str, Any] = {"model": model}
    if api_key is not None:
        kwargs["api_key"] = api_key
    return Anthropic(**kwargs)
