from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..memory import estimate_model_memory


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
