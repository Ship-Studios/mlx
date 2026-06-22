from __future__ import annotations


class _HardwareInfoCanRunMlxMixin:
    @property
    def can_run_mlx(self) -> bool:
        """Whether mlx-lm can actually execute here (Apple-silicon Mac)."""
        return self.system == "Darwin" and self.is_apple_silicon
