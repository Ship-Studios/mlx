from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class Check:
    """One readiness check outcome."""

    name: str
    status: str  # OK | WARN | FAIL
    detail: str
    remediation: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "status": self.status,
            "detail": self.detail,
            "remediation": self.remediation,
        }
