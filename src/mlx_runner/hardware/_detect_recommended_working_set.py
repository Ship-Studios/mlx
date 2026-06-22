from __future__ import annotations

from typing import Optional


def _detect_recommended_working_set(
    device_info: Optional[dict], total_ram: Optional[int]
) -> Optional[int]:
    if device_info:
        wss = device_info.get("max_recommended_working_set_size")
        if wss:
            try:
                return int(wss)
            except (TypeError, ValueError):
                pass
    # Heuristic: Apple recommends keeping roughly 75% of unified memory for the
    # GPU working set; the rest is held back for the OS and other processes.
    if total_ram:
        return int(total_ram * 0.75)
    return None
