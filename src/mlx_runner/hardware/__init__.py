"""Detect the local machine's capabilities relevant to running LLMs with MLX.

Everything here is best-effort and degrades gracefully: on a non-macOS machine,
or when ``sysctl`` / ``mlx`` are unavailable, fields are simply ``None`` rather
than raising. The module imports cleanly without ``mlx`` installed.
"""
from ._sysctl import _sysctl
from ._sysctl_int import _sysctl_int
from ._mlx_available import _mlx_available
from .hardware_info import HardwareInfo
from ._detect_total_ram import _detect_total_ram
from ._query_metal_device_info import _query_metal_device_info
from ._detect_gpu_cores import _detect_gpu_cores
from ._detect_recommended_working_set import _detect_recommended_working_set
from .detect_hardware import detect_hardware

__all__ = [
    "_sysctl",
    "_sysctl_int",
    "_mlx_available",
    "HardwareInfo",
    "_detect_total_ram",
    "_query_metal_device_info",
    "_detect_gpu_cores",
    "_detect_recommended_working_set",
    "detect_hardware",
]
