from __future__ import annotations

from typing import List, Optional

from .anthropic_error import AnthropicError
from .parsed_request import ParsedRequest
from ._content_to_text import _content_to_text


def parse_request(body: dict) -> ParsedRequest:
    """Validate and normalize a decoded ``/v1/messages`` JSON body."""
    if not isinstance(body, dict):
        raise AnthropicError(400, "invalid_request_error", "request body must be a JSON object.")

    model = body.get("model")
    if not isinstance(model, str) or not model:
        raise AnthropicError(400, "invalid_request_error", "`model` is required and must be a string.")

    max_tokens = body.get("max_tokens")
    if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens <= 0:
        raise AnthropicError(400, "invalid_request_error", "`max_tokens` is required and must be a positive integer.")

    raw_messages = body.get("messages")
    if not isinstance(raw_messages, list) or not raw_messages:
        raise AnthropicError(400, "invalid_request_error", "`messages` is required and must be a non-empty array.")

    messages: List[dict] = []
    system_from_messages: List[str] = []
    for i, m in enumerate(raw_messages):
        if not isinstance(m, dict):
            raise AnthropicError(400, "invalid_request_error", f"messages[{i}] must be an object.")
        role = m.get("role")
        # The Anthropic schema only allows user/assistant in `messages`, but Claude
        # Code includes its session/hook context as a `system`-role message in the
        # array. Fold any system-role message into the system prompt rather than 400.
        if role == "system":
            system_from_messages.append(_content_to_text(m.get("content"), where=f"messages[{i}]"))
            continue
        if role not in ("user", "assistant"):
            raise AnthropicError(
                400, "invalid_request_error", f"messages[{i}].role must be 'user' or 'assistant'."
            )
        text = _content_to_text(m.get("content"), where=f"messages[{i}]")
        messages.append({"role": role, "content": text})

    system_val = body.get("system")
    system = _content_to_text(system_val, where="system") if system_val is not None else None
    if system_from_messages:
        parts = ([system] if system else []) + system_from_messages
        system = "\n\n".join(p for p in parts if p) or None

    stops = body.get("stop_sequences") or []
    if not isinstance(stops, list) or any(not isinstance(s, str) for s in stops):
        raise AnthropicError(400, "invalid_request_error", "`stop_sequences` must be an array of strings.")

    def _num(name, cast):
        v = body.get(name)
        if v is None:
            return None
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise AnthropicError(400, "invalid_request_error", f"`{name}` must be a number.")
        return cast(v)

    return ParsedRequest(
        model=model,
        max_tokens=max_tokens,
        messages=messages,
        system=system,
        temperature=_num("temperature", float),
        top_p=_num("top_p", float),
        top_k=_num("top_k", int),
        stop_sequences=[s for s in stops if s],
        stream=bool(body.get("stream", False)),
    )
