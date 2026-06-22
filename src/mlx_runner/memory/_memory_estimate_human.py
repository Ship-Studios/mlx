from __future__ import annotations

from .format_bytes import format_bytes


class _MemoryEstimateHumanMixin:
    def human(self) -> str:
        return (
            f"weights={format_bytes(self.weights_bytes)}, "
            f"kv_cache={format_bytes(self.kv_cache_bytes)}, "
            f"overhead={format_bytes(self.overhead_bytes)}, "
            f"total={format_bytes(self.total_bytes)}"
        )
