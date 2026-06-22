from __future__ import annotations

from typing import Optional


def _detect_gpu_cores(device_info: Optional[dict]) -> Optional[int]:
    if not device_info:
        return None
    for key in ("gpu_core_count", "num_gpu_cores", "gpu_cores"):
        if key in device_info:
            try:
                return int(device_info[key])
            except (TypeError, ValueError):
                return None
    return None
