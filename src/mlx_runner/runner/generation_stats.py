from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class GenerationStats:
    """Timing and memory statistics reported by mlx-lm after a generation."""

    prompt_tokens: int = 0
    generation_tokens: int = 0
    prompt_tps: float = 0.0
    generation_tps: float = 0.0
    peak_memory_gb: float = 0.0
    finish_reason: Optional[str] = None

    def human(self) -> str:
        return (
            f"prompt {self.prompt_tokens} tok @ {self.prompt_tps:.1f} tok/s | "
            f"gen {self.generation_tokens} tok @ {self.generation_tps:.1f} tok/s | "
            f"peak {self.peak_memory_gb:.2f} GB"
            + (f" | {self.finish_reason}" if self.finish_reason else "")
        )
