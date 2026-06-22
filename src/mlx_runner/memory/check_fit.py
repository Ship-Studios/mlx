from __future__ import annotations

from .memory_estimate import MemoryEstimate
from .fit_result import FitResult


def check_fit(estimate, available_bytes: int, safety_fraction: float = 0.9) -> FitResult:
    """Check whether ``estimate`` fits inside ``safety_fraction`` of available memory.

    ``estimate`` may be a :class:`MemoryEstimate` or a raw byte count.
    """
    if not 0 < safety_fraction <= 1:
        raise ValueError("safety_fraction must be in (0, 1]")
    required = (
        estimate.total_bytes if isinstance(estimate, MemoryEstimate) else int(estimate)
    )
    budget = int(available_bytes * safety_fraction)
    return FitResult(
        required_bytes=required,
        available_bytes=int(available_bytes),
        budget_bytes=budget,
        safety_fraction=safety_fraction,
        fits=required <= budget,
        headroom_bytes=budget - required,
    )
