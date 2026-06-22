from __future__ import annotations

from dataclasses import dataclass

from ._fit_result_human import _FitResultHumanMixin
from ._fit_result_to_dict import _FitResultToDictMixin


@dataclass
class FitResult(_FitResultHumanMixin, _FitResultToDictMixin):
    """Result of checking an estimate against an available-memory budget."""

    required_bytes: int
    available_bytes: int
    budget_bytes: int
    safety_fraction: float
    fits: bool
    headroom_bytes: int
