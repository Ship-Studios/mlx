from __future__ import annotations

from .parsed_request import ParsedRequest


def _generation_config(parsed: ParsedRequest):
    from ..runner import GenerationConfig

    kwargs = {"max_tokens": parsed.max_tokens}
    if parsed.temperature is not None:
        kwargs["temperature"] = parsed.temperature
    if parsed.top_p is not None:
        kwargs["top_p"] = parsed.top_p
    if parsed.top_k is not None:
        kwargs["top_k"] = parsed.top_k
    return GenerationConfig(**kwargs)
