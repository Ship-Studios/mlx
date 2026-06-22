"""A small curated catalog of ready-to-run mlx-community models.

Used by the ``setup`` command to recommend a model that fits the local machine.
Pure-Python and dependency-free: the memory math reuses :mod:`mlx_runner.memory`,
so recommendations work on any platform without mlx, a network, or a download.

Parameter counts are approximate (rounded from each model card) — they only need
to be good enough to rank models and check fit, not to be exact.
"""
from ._constants import CATALOG
from .catalog_model import CatalogModel
from .fitting_models import fitting_models
from .recommend_model import recommend_model

__all__ = [
    "CatalogModel",
    "CATALOG",
    "fitting_models",
    "recommend_model",
]
