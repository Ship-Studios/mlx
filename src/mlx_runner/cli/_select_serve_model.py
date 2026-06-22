from __future__ import annotations

import sys
from typing import Optional

from ..catalog import fitting_models
from ..memory import format_bytes


def _select_serve_model(hw, *, safety_fraction: float = 0.8) -> Optional[str]:
    """Pick a catalog model that is sized to fit this machine, for `serve` with no --model.

    Lists every catalog model whose weight estimate fits the detected memory budget
    (smallest first) and lets the user choose. The default is the most capable model
    that fits — pick a smaller one for more headroom. With no TTY (headless cloud box
    or a tunnel) it silently takes that default so serving never blocks on input.

    Returns the chosen Hugging Face repo id, or ``None`` if memory can't be determined
    or nothing in the catalog fits (the caller should treat ``None`` as a failure).
    """
    available = hw.recommended_working_set_bytes or hw.total_ram_bytes
    if not available:
        print(
            "error: could not determine available memory to pick a model; pass --model.",
            file=sys.stderr,
        )
        return None

    fitting = fitting_models(available, safety_fraction=safety_fraction)
    if not fitting:
        print(
            "error: no catalog model fits this machine's memory budget. "
            "Pass a tiny model explicitly with --model.",
            file=sys.stderr,
        )
        return None

    default = fitting[-1]  # largest that fits == the most capable safe choice
    default_idx = len(fitting)  # 1-based index of the default

    print("Models that fit this machine (most capable that fits is the default):", file=sys.stderr)
    for i, m in enumerate(fitting, 1):
        marker = "  [default]" if m is default else ""
        print(
            f"  {i}) {m.name}  (~{m.params / 1e9:.1f}B @ {m.quant_bits}-bit, "
            f"weights {format_bytes(m.estimate().weights_bytes)}){marker}",
            file=sys.stderr,
        )

    if not sys.stdin.isatty():
        print(f"No TTY; auto-selecting {default.name} ({default.repo_id}).", file=sys.stderr)
        return default.repo_id

    while True:
        try:
            raw = input(f"Choose a model [{default_idx}]: ").strip()
        except EOFError:
            return default.repo_id
        if not raw:
            return default.repo_id
        if raw.isdigit() and 1 <= int(raw) <= len(fitting):
            return fitting[int(raw) - 1].repo_id
        print(f"  enter 1-{len(fitting)}, or press Enter for the default.", file=sys.stderr)
