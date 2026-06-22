from __future__ import annotations

from .anthropic_error import AnthropicError


def _content_to_text(content, *, where: str) -> str:
    """Flatten a message/system ``content`` (string or block list) to plain text.

    Rejects non-text blocks (images, tool_use, …) with a 400 — a local text model
    can't honor them, and silently dropping them would corrupt the conversation.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "text":
                btype = block.get("type") if isinstance(block, dict) else type(block).__name__
                raise AnthropicError(
                    400, "invalid_request_error",
                    f"{where}: only text content blocks are supported, got {btype!r}.",
                )
            parts.append(str(block.get("text", "")))
        return "".join(parts)
    raise AnthropicError(
        400, "invalid_request_error", f"{where}: content must be a string or a list of text blocks."
    )
