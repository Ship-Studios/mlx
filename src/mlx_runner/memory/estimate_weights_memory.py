from __future__ import annotations

from typing import Optional

from .bytes_per_weight import bytes_per_weight


def estimate_weights_memory(
    num_params: int,
    quant_bits: Optional[int] = None,
    dtype: str = "float16",
    group_size: int = 64,
) -> int:
    """Bytes required to hold the model weights."""
    if num_params <= 0:
        raise ValueError("num_params must be positive")
    return int(num_params * bytes_per_weight(quant_bits, dtype, group_size))
