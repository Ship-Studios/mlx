from __future__ import annotations

import os
import platform
from typing import Optional

import mlx_runner.hardware as _pkg
from ._mlx_available import _mlx_available
from ._sysctl import _sysctl
from ._sysctl_int import _sysctl_int
from .hardware_info import HardwareInfo


def detect_hardware() -> HardwareInfo:
    """Probe the current machine and return a :class:`HardwareInfo`."""
    system = platform.system()
    machine = platform.machine()
    is_apple_silicon = system == "Darwin" and machine == "arm64"

    chip: Optional[str] = None
    perf = eff = cpu = None
    os_version: Optional[str] = None

    if system == "Darwin":
        chip = _sysctl("machdep.cpu.brand_string")
        # On Apple silicon, perflevel0 is the performance (P) cluster and
        # perflevel1 the efficiency (E) cluster.
        perf = _sysctl_int("hw.perflevel0.logicalcpu")
        eff = _sysctl_int("hw.perflevel1.logicalcpu")
        cpu = _sysctl_int("hw.logicalcpu") or _sysctl_int("hw.ncpu") or os.cpu_count()
        mac_ver = platform.mac_ver()[0]
        if mac_ver:
            os_version = f"macOS {mac_ver}"
    else:
        cpu = os.cpu_count()
        os_version = f"{system} {platform.release()}".strip()

    mlx_available = _mlx_available()
    device_info = _pkg._query_metal_device_info() if mlx_available else None
    total_ram = _pkg._detect_total_ram()

    return _pkg.HardwareInfo(
        system=system,
        machine=machine,
        is_apple_silicon=is_apple_silicon,
        chip=chip,
        os_version=os_version,
        cpu_cores=cpu,
        performance_cores=perf,
        efficiency_cores=eff,
        gpu_cores=_pkg._detect_gpu_cores(device_info),
        total_ram_bytes=total_ram,
        mlx_available=mlx_available,
        recommended_working_set_bytes=_pkg._detect_recommended_working_set(
            device_info, total_ram
        ),
    )
