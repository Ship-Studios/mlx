"""Readiness checks for running LLMs locally with mlx-lm.

``run_checks`` returns a list of :class:`Check` results so the CLI (and tests)
can render them and decide an exit status. Everything is best-effort and never
raises; a check that can't be evaluated degrades to ``warn``.
"""
from __future__ import annotations

import shutil

from ._constants import FAIL, OK, WARN, _MIN_USABLE_RAM
from ._module_available import _module_available
from .check import Check
from .is_ready import is_ready
from .run_checks import run_checks
from .worst_status import worst_status

__all__ = [
    "OK",
    "WARN",
    "FAIL",
    "_MIN_USABLE_RAM",
    "Check",
    "_module_available",
    "run_checks",
    "worst_status",
    "is_ready",
]
