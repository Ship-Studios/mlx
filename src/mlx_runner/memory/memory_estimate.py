from __future__ import annotations

from dataclasses import dataclass

from ._memory_estimate_total_bytes import _MemoryEstimateTotalBytesMixin
from ._memory_estimate_human import _MemoryEstimateHumanMixin
from ._memory_estimate_to_dict import _MemoryEstimateToDictMixin


@dataclass
class MemoryEstimate(
    _MemoryEstimateTotalBytesMixin,
    _MemoryEstimateHumanMixin,
    _MemoryEstimateToDictMixin,
):
    """Breakdown of estimated memory for running a model."""

    num_params: int
    weights_bytes: int
    kv_cache_bytes: int
    overhead_bytes: int
