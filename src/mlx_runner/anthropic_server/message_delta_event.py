from __future__ import annotations

from typing import Optional

from .sse_event import sse_event


def message_delta_event(stop_reason: str, stop_sequence: Optional[str], output_tokens: int) -> str:
    return sse_event(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": stop_sequence},
            "usage": {"output_tokens": output_tokens},
        },
    )
