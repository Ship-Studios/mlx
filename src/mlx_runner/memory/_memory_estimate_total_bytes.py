from __future__ import annotations


class _MemoryEstimateTotalBytesMixin:
    @property
    def total_bytes(self) -> int:
        return self.weights_bytes + self.kv_cache_bytes + self.overhead_bytes
