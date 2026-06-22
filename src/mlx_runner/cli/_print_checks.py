from __future__ import annotations

from typing import List

from ..doctor import OK, Check
from ._constants import _STATUS_MARK


def _print_checks(checks: List[Check]) -> None:
    for c in checks:
        mark = _STATUS_MARK.get(c.status, "?")
        print(f"  {mark} {c.name}: {c.detail}")
        if c.status != OK and c.remediation:
            print(f"      → {c.remediation}")
