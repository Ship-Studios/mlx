from __future__ import annotations

from typing import Optional

from ._constants import _DTYPE_BYTES


def bytes_per_weight(
    quant_bits: Optional[int] = None,
    dtype: str = "float16",
    group_size: int = 64,
) -> float:
    """Average bytes stored per weight parameter.

    For quantized weights, MLX stores ``quant_bits`` per weight plus one fp16
    scale and one fp16 bias per group of ``group_size`` weights (4 extra bytes
    per group). For unquantized weights it is just the dtype width.
    """
    if quant_bits is not None:
        if quant_bits <= 0:
            raise ValueError("quant_bits must be positive")
        if group_size <= 0:
            raise ValueError("group_size must be positive")
        return quant_bits / 8.0 + 4.0 / group_size
    key = dtype.lower()
    if key not in _DTYPE_BYTES:
        raise ValueError(f"unknown dtype {dtype!r}; known: {sorted(_DTYPE_BYTES)}")
    return _DTYPE_BYTES[key]
