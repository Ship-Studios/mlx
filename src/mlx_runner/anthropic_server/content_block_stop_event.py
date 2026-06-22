from __future__ import annotations

from .sse_event import sse_event


def content_block_stop_event() -> str:
    return sse_event("content_block_stop", {"type": "content_block_stop", "index": 0})
