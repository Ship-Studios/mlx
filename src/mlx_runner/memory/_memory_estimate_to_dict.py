from __future__ import annotations

from dataclasses import asdict


class _MemoryEstimateToDictMixin:
    def to_dict(self) -> dict:
        d = asdict(self)
        d["total_bytes"] = self.total_bytes
        return d
