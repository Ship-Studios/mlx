"""A small curated catalog of ready-to-run mlx-community models.

Used by the ``setup`` command to recommend a model that fits the local machine.
Pure-Python and dependency-free: the memory math reuses :mod:`mlx_runner.memory`,
so recommendations work on any platform without mlx, a network, or a download.

Parameter counts are approximate (rounded from each model card) — they only need
to be good enough to rank models and check fit, not to be exact.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from .memory import check_fit, estimate_model_memory


@dataclass(frozen=True)
class CatalogModel:
    """A pre-quantized model build with enough metadata to size it."""

    name: str  # human label, e.g. "Qwen2.5 7B Instruct"
    repo_id: str  # Hugging Face repo id
    params: int  # approximate parameter count
    quant_bits: Optional[int] = 4  # None == full precision

    def estimate(self):
        """A :class:`mlx_runner.memory.MemoryEstimate` for this build's weights."""
        return estimate_model_memory(self.params, quant_bits=self.quant_bits)


_B = 1_000_000_000

# Smallest to largest. All are 4-bit instruct builds from the mlx-community org.
CATALOG: Tuple[CatalogModel, ...] = (
    CatalogModel("Llama 3.2 1B Instruct", "mlx-community/Llama-3.2-1B-Instruct-4bit", int(1.24 * _B)),
    CatalogModel("Qwen2.5 1.5B Instruct", "mlx-community/Qwen2.5-1.5B-Instruct-4bit", int(1.54 * _B)),
    CatalogModel("Llama 3.2 3B Instruct", "mlx-community/Llama-3.2-3B-Instruct-4bit", int(3.21 * _B)),
    CatalogModel("Qwen2.5 7B Instruct", "mlx-community/Qwen2.5-7B-Instruct-4bit", int(7.62 * _B)),
    CatalogModel("Llama 3.1 8B Instruct", "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit", int(8.03 * _B)),
    CatalogModel("Qwen2.5 14B Instruct", "mlx-community/Qwen2.5-14B-Instruct-4bit", int(14.8 * _B)),
    CatalogModel("Qwen2.5 32B Instruct", "mlx-community/Qwen2.5-32B-Instruct-4bit", int(32.5 * _B)),
    CatalogModel("Qwen2.5 72B Instruct", "mlx-community/Qwen2.5-72B-Instruct-4bit", int(72.7 * _B)),
)


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


def recommend_model(
    available_bytes: int,
    safety_fraction: float = 0.9,
    catalog: Tuple[CatalogModel, ...] = CATALOG,
) -> Optional[CatalogModel]:
    """The largest catalog model that fits the budget, or ``None`` if none fit."""
    fitting = fitting_models(available_bytes, safety_fraction, catalog)
    return fitting[-1] if fitting else None
