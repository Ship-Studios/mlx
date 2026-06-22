from __future__ import annotations

from typing import List, Optional


class _StopSequenceFilterInitMixin:
    def __init__(self, stops: List[str]):
        self.stops = [s for s in stops if s]
        self.max_len = max((len(s) for s in self.stops), default=0)
        self.buf = ""
        self.matched: Optional[str] = None
