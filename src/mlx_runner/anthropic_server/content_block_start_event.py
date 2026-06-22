from __future__ import annotations

from .sse_event import sse_event


def content_block_start_event() -> str:
    return sse_event(
        "content_block_start",
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
    )
