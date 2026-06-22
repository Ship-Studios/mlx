from __future__ import annotations

from typing import Optional

from .estimate_weights_memory import estimate_weights_memory
from .estimate_kv_cache_memory import estimate_kv_cache_memory
from .memory_estimate import MemoryEstimate


def estimate_model_memory(
    num_params: int,
    quant_bits: Optional[int] = None,
    dtype: str = "float16",
    group_size: int = 64,
    *,
    num_layers: Optional[int] = None,
    num_kv_heads: Optional[int] = None,
    head_dim: Optional[int] = None,
    seq_len: int = 4096,
    kv_bits: Optional[int] = None,
    batch_size: int = 1,
    overhead_fraction: float = 0.05,
) -> MemoryEstimate:
    """Estimate total memory to load weights plus (optionally) a KV cache.

    The KV-cache term is only included when ``num_layers``, ``num_kv_heads`` and
    ``head_dim`` are all provided; otherwise it is zero. ``overhead_fraction``
    accounts for activations, framework buffers, and fragmentation.
    """
    weights = estimate_weights_memory(num_params, quant_bits, dtype, group_size)
    kv = 0
    if num_layers and num_kv_heads and head_dim:
        kv = estimate_kv_cache_memory(
            num_layers, num_kv_heads, head_dim, seq_len, kv_bits, batch_size
        )
    overhead = int(weights * overhead_fraction)
    return MemoryEstimate(
        num_params=num_params,
        weights_bytes=weights,
        kv_cache_bytes=kv,
        overhead_bytes=overhead,
    )
