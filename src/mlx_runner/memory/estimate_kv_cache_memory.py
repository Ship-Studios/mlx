from __future__ import annotations

from typing import Optional


def estimate_kv_cache_memory(
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    seq_len: int,
    kv_bits: Optional[int] = None,
    batch_size: int = 1,
) -> int:
    """Bytes for the key/value cache at a given sequence length.

    Stores both K and V: ``2 * layers * kv_heads * head_dim * seq_len * batch``
    elements. Defaults to fp16 (2 bytes); pass ``kv_bits`` for a quantized cache.
    """
    if min(num_layers, num_kv_heads, head_dim, seq_len, batch_size) <= 0:
        raise ValueError("kv-cache dimensions must all be positive")
    bytes_per_elem = (kv_bits / 8.0) if kv_bits else 2.0
    elems = 2 * num_layers * num_kv_heads * head_dim * seq_len * batch_size
    return int(elems * bytes_per_elem)
