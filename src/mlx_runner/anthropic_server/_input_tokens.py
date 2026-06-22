from __future__ import annotations

from .parsed_request import ParsedRequest
from ._count_tokens import _count_tokens


def _input_tokens(runner, parsed: ParsedRequest) -> int:
    parts = [parsed.system or ""] + [m["content"] for m in parsed.messages]
    return _count_tokens(runner, "\n".join(p for p in parts if p))
