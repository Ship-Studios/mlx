from __future__ import annotations

from typing import List

from ._constants import FAIL, OK, WARN
from .check import Check


def worst_status(checks: List[Check]) -> str:
    """The most severe status across ``checks`` (FAIL > WARN > OK)."""
    statuses = {c.status for c in checks}
    if FAIL in statuses:
        return FAIL
    if WARN in statuses:
        return WARN
    return OK
