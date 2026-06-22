from __future__ import annotations

from typing import Optional


def build_message(
    *,
    message_id: str,
    model: str,
    text: str,
    stop_reason: str,
    stop_sequence: Optional[str],
    input_tokens: int,
    output_tokens: int,
) -> dict:
    """A complete non-streaming ``Message`` object."""
    return {
        "id": message_id,
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": text}],
        "stop_reason": stop_reason,
        "stop_sequence": stop_sequence,
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
    }
