from __future__ import annotations

import json
import sys

from ..hardware import detect_hardware
from ..memory import format_bytes


def cmd_info(args) -> int:
    hw = detect_hardware()
    if args.json:
        print(json.dumps(hw.to_dict(), indent=2))
        return 0
    print(f"System:            {hw.system} ({hw.machine})")
    print(f"Chip:              {hw.chip or 'unknown'}")
    print(f"OS:                {hw.os_version or 'unknown'}")
    cores = hw.cpu_cores or "?"
    pe = ""
    if hw.performance_cores or hw.efficiency_cores:
        pe = f" ({hw.performance_cores or '?'}P + {hw.efficiency_cores or '?'}E)"
    print(f"CPU cores:         {cores}{pe}")
    print(f"GPU cores:         {hw.gpu_cores if hw.gpu_cores is not None else 'unknown'}")
    print(f"Total RAM:         {format_bytes(hw.total_ram_bytes)}")
    print(f"Recommended budget:{format_bytes(hw.recommended_working_set_bytes)}")
    print(f"MLX importable:    {hw.mlx_available}")
    print(f"Can run mlx-lm:    {hw.can_run_mlx}")
    if not hw.can_run_mlx:
        print(
            "\n⚠  This is not an Apple-silicon Mac; mlx-lm cannot execute here.",
            file=sys.stderr,
        )
    return 0
