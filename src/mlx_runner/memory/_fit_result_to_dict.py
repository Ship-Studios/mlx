from __future__ import annotations

from dataclasses import asdict


class _FitResultToDictMixin:
    def to_dict(self) -> dict:
        return asdict(self)
