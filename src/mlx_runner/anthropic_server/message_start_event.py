from __future__ import annotations

from .sse_event import sse_event


def message_start_event(message_id: str, model: str, input_tokens: int) -> str:
    msg = {
        "id": message_id,
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [],
        "stop_reason": None,
        "stop_sequence": None,
        "usage": {"input_tokens": input_tokens, "output_tokens": 0},
    }
    return sse_event("message_start", {"type": "message_start", "message": msg})
