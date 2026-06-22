from __future__ import annotations

from dataclasses import asdict


class _HardwareInfoToDictMixin:
    def to_dict(self) -> dict:
        return asdict(self)
