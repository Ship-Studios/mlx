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
        """A :class:`mlx_runner.memory.MemoryEstimate` for this build.

        Catalog entries carry no architecture dims (layers/kv-heads/head-dim), so
        an exact KV-cache term can't be computed here. Instead we widen the
        ``overhead_fraction`` to ~25% to reserve room for activations, the Metal
        allocator, and a working KV cache — the 5% default only covers weights and
        would let ``recommend_model`` pick a build that over-commits memory and
        panics the GPU driver at load time (see ``runner._apply_memory_guard``).
        """
        return estimate_model_memory(
            self.params, quant_bits=self.quant_bits, overhead_fraction=0.25
        )
