from __future__ import annotations

from .format_bytes import format_bytes


class _FitResultHumanMixin:
    def human(self) -> str:
        verdict = "FITS" if self.fits else "DOES NOT FIT"
        sign = "" if self.headroom_bytes < 0 else "+"
        return (
            f"{verdict}: needs {format_bytes(self.required_bytes)} vs budget "
            f"{format_bytes(self.budget_bytes)} "
            f"({int(self.safety_fraction * 100)}% of {format_bytes(self.available_bytes)}); "
            f"headroom {sign}{format_bytes(self.headroom_bytes)}"
        )
