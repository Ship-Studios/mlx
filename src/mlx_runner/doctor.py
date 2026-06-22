"""Readiness checks for running LLMs locally with mlx-lm.

``run_checks`` returns a list of :class:`Check` results so the CLI (and tests)
can render them and decide an exit status. Everything is best-effort and never
raises; a check that can't be evaluated degrades to ``warn``.
"""
from __future__ import annotations

import importlib.util
import platform
import shutil
import sys
from dataclasses import dataclass
from typing import List, Optional

from .hardware import HardwareInfo, detect_hardware
from .memory import format_bytes

OK = "ok"
WARN = "warn"
FAIL = "fail"

# Minimum unified memory below which even the smallest 4-bit build is a squeeze.
_MIN_USABLE_RAM = 6 * 1024 ** 3


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


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def run_checks(hw: Optional[HardwareInfo] = None) -> List[Check]:
    """Evaluate the environment and return a list of :class:`Check`."""
    hw = hw or detect_hardware()
    checks: List[Check] = []

    # 1. Apple silicon — the hard requirement for mlx.
    if hw.is_apple_silicon:
        checks.append(Check("Apple silicon", OK, f"{hw.chip or 'arm64 Mac'}"))
    elif hw.system == "Darwin":
        checks.append(Check(
            "Apple silicon", FAIL, "Intel Mac detected",
            "mlx requires an Apple-silicon Mac (M1 or newer).",
        ))
    else:
        checks.append(Check(
            "Apple silicon", FAIL, f"non-macOS host ({hw.system})",
            "mlx-lm only runs on macOS with Apple silicon.",
        ))

    # 2. Python version.
    v = sys.version_info
    pyver = f"{v.major}.{v.minor}.{v.micro}"
    if (v.major, v.minor) >= (3, 8):
        checks.append(Check("Python", OK, f"{pyver}"))
    else:
        checks.append(Check(
            "Python", FAIL, f"{pyver}",
            "Python 3.8+ is required; install a newer Python.",
        ))

    # 3. mlx importable.
    if hw.mlx_available:
        checks.append(Check("mlx", OK, "importable"))
    else:
        checks.append(Check(
            "mlx", WARN, "not importable",
            "Installed automatically with mlx-lm: `pip install mlx-lm`.",
        ))

    # 4. mlx-lm importable — needed to actually load/run a model.
    if _module_available("mlx_lm"):
        checks.append(Check("mlx-lm", OK, "importable"))
    else:
        checks.append(Check(
            "mlx-lm", FAIL, "not installed",
            "Install it with `pip install mlx-lm` (Apple silicon).",
        ))

    # 5. Memory.
    ram = hw.total_ram_bytes
    if ram is None:
        checks.append(Check("Memory", WARN, "could not determine RAM"))
    elif ram >= _MIN_USABLE_RAM:
        budget = hw.recommended_working_set_bytes or ram
        checks.append(Check(
            "Memory", OK, f"{format_bytes(ram)} ({format_bytes(budget)} usable)",
        ))
    else:
        checks.append(Check(
            "Memory", WARN, f"{format_bytes(ram)} is low",
            "Stick to ~1B 4-bit models; larger ones may not fit.",
        ))

    # 6. cloudflared — optional, only needed for `serve --tunnel`. Never fails.
    if shutil.which("cloudflared"):
        checks.append(Check("cloudflared", OK, "installed (enables `serve --tunnel`)"))
    else:
        checks.append(Check(
            "cloudflared", OK,
            "not installed — only needed for `serve --tunnel` (brew install cloudflared)",
        ))

    return checks


def worst_status(checks: List[Check]) -> str:
    """The most severe status across ``checks`` (FAIL > WARN > OK)."""
    statuses = {c.status for c in checks}
    if FAIL in statuses:
        return FAIL
    if WARN in statuses:
        return WARN
    return OK


def is_ready(checks: List[Check]) -> bool:
    """True when no check failed (warnings are tolerated)."""
    return worst_status(checks) != FAIL
