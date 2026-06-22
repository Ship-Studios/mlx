from __future__ import annotations

from typing import Optional, Tuple

# Average bytes per parameter for common unquantized dtypes.
_DTYPE_BYTES = {
    "float32": 4.0, "fp32": 4.0, "f32": 4.0,
    "float16": 2.0, "fp16": 2.0, "f16": 2.0,
    "bfloat16": 2.0, "bf16": 2.0,
    "float64": 8.0, "fp64": 8.0,
}

# Highest-quality-first; used by recommend_quantization. ``None`` means no
# quantization (full ``dtype`` precision).
_DEFAULT_QUANT_CANDIDATES: Tuple[Optional[int], ...] = (None, 8, 6, 4, 3, 2)
