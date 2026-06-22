from __future__ import annotations

from typing import Optional, Tuple

from ._constants import _DEFAULT_QUANT_CANDIDATES
from .estimate_weights_memory import estimate_weights_memory


def recommend_quantization(
    num_params: int,
    available_bytes: int,
    dtype: str = "float16",
    group_size: int = 64,
    safety_fraction: float = 0.9,
    candidates: Tuple[Optional[int], ...] = _DEFAULT_QUANT_CANDIDATES,
) -> dict:
    """Recommend the highest-quality precision whose weights fit the budget.

    Returns a dict with ``quant_bits`` (``None`` == full precision),
    ``weights_bytes`` and ``fits``. If nothing fits, returns the smallest
    candidate with ``fits=False``.
    """
    budget = int(available_bytes * safety_fraction)
    best_small = None
    for bits in candidates:
        weights = estimate_weights_memory(num_params, bits, dtype, group_size)
        best_small = {"quant_bits": bits, "weights_bytes": weights, "fits": weights <= budget}
        if weights <= budget:
            return best_small
    return best_small or {"quant_bits": None, "weights_bytes": 0, "fits": False}
