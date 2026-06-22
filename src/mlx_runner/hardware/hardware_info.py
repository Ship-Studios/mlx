from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ._hardware_info_can_run_mlx import _HardwareInfoCanRunMlxMixin
from ._hardware_info_to_dict import _HardwareInfoToDictMixin


@dataclass
class HardwareInfo(_HardwareInfoCanRunMlxMixin, _HardwareInfoToDictMixin):
    """A snapshot of the host's relevant hardware/software characteristics."""

    system: str  # 'Darwin', 'Linux', ...
    machine: str  # 'arm64', 'x86_64', ...
    is_apple_silicon: bool
    chip: Optional[str]  # e.g. 'Apple M3 Pro'
    os_version: Optional[str]  # e.g. 'macOS 15.5'
    cpu_cores: Optional[int]
    performance_cores: Optional[int]
    efficiency_cores: Optional[int]
    gpu_cores: Optional[int]
    total_ram_bytes: Optional[int]
    mlx_available: bool
    recommended_working_set_bytes: Optional[int]
