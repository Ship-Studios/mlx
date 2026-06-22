"""Detect the local machine's capabilities relevant to running LLMs with MLX.

Everything here is best-effort and degrades gracefully: on a non-macOS machine,
or when ``sysctl`` / ``mlx`` are unavailable, fields are simply ``None`` rather
than raising. The module imports cleanly without ``mlx`` installed.
"""
from __future__ import annotations

import importlib.util
import os
import platform
import subprocess
from dataclasses import asdict, dataclass
from typing import Optional


def _sysctl(key: str) -> Optional[str]:
    """Return a sysctl value as text, or ``None`` if unavailable."""
    try:
        out = subprocess.run(
            ["sysctl", "-n", key],
            capture_output=True,
            text=True,
            timeout=2,
            check=True,
        )
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return None
    val = out.stdout.strip()
    return val or None


def _sysctl_int(key: str) -> Optional[int]:
    val = _sysctl(key)
    if val is None:
        return None
    try:
        return int(val)
    except ValueError:
        return None


def _mlx_available() -> bool:
    """True if the ``mlx`` package is importable (does not import it)."""
    try:
        return importlib.util.find_spec("mlx") is not None
    except (ImportError, ValueError):
        return False


@dataclass
class HardwareInfo:
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

    @property
    def can_run_mlx(self) -> bool:
        """Whether mlx-lm can actually execute here (Apple-silicon Mac)."""
        return self.system == "Darwin" and self.is_apple_silicon

    def to_dict(self) -> dict:
        return asdict(self)


def _detect_total_ram() -> Optional[int]:
    # macOS / BSD expose the physical memory size via sysctl.
    v = _sysctl_int("hw.memsize")
    if v:
        return v
    # POSIX fallback (Linux and friends).
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return None


def _query_metal_device_info() -> Optional[dict]:
    """Return mlx's metal device_info dict, or ``None`` if unavailable."""
    try:
        import mlx.core as mx

        info = mx.metal.device_info()
        return dict(info) if info else None
    except Exception:  # pragma: no cover - depends on hardware/mlx version
        return None


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
    device_info = _query_metal_device_info() if mlx_available else None
    total_ram = _detect_total_ram()

    return HardwareInfo(
        system=system,
        machine=machine,
        is_apple_silicon=is_apple_silicon,
        chip=chip,
        os_version=os_version,
        cpu_cores=cpu,
        performance_cores=perf,
        efficiency_cores=eff,
        gpu_cores=_detect_gpu_cores(device_info),
        total_ram_bytes=total_ram,
        mlx_available=mlx_available,
        recommended_working_set_bytes=_detect_recommended_working_set(
            device_info, total_ram
        ),
    )
