from __future__ import annotations

from typing import List

from ._constants import FAIL
from .check import Check
from .worst_status import worst_status


def is_ready(checks: List[Check]) -> bool:
    """True when no check failed (warnings are tolerated)."""
    return worst_status(checks) != FAIL
