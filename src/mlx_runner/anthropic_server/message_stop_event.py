from __future__ import annotations

from .sse_event import sse_event


def message_stop_event() -> str:
    return sse_event("message_stop", {"type": "message_stop"})
