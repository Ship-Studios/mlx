from __future__ import annotations

from .sse_event import sse_event


def ping_event() -> str:
    return sse_event("ping", {"type": "ping"})
