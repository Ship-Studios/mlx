from __future__ import annotations

from typing import Optional, Tuple

from ._constants import CATALOG
from .catalog_model import CatalogModel
from .fitting_models import fitting_models


def recommend_model(
    available_bytes: int,
    safety_fraction: float = 0.9,
    catalog: Tuple[CatalogModel, ...] = CATALOG,
) -> Optional[CatalogModel]:
    """The largest catalog model that fits the budget, or ``None`` if none fit."""
    fitting = fitting_models(available_bytes, safety_fraction, catalog)
    return fitting[-1] if fitting else None
