from __future__ import annotations

from typing import List, Tuple

from ..memory import check_fit
from ._constants import CATALOG
from .catalog_model import CatalogModel


def fitting_models(
    available_bytes: int,
    safety_fraction: float = 0.9,
    catalog: Tuple[CatalogModel, ...] = CATALOG,
) -> List[CatalogModel]:
    """All catalog models whose weight estimate fits the budget, smallest first."""
    out = []
    for m in sorted(catalog, key=lambda m: m.params):
        if check_fit(m.estimate(), available_bytes, safety_fraction).fits:
            out.append(m)
    return out
