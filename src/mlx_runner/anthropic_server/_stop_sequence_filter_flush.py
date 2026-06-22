from __future__ import annotations


class _StopSequenceFilterFlushMixin:
    def flush(self) -> str:
        out, self.buf = self.buf, ""
        return out
