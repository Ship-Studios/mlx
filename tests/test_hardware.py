from mlx_runner import hardware
from mlx_runner.hardware import HardwareInfo, detect_hardware


def test_detect_hardware_returns_sane_info():
    hw = detect_hardware()
    assert isinstance(hw, HardwareInfo)
    assert isinstance(hw.system, str) and hw.system
    assert isinstance(hw.machine, str) and hw.machine
    assert isinstance(hw.is_apple_silicon, bool)
    assert isinstance(hw.mlx_available, bool)
    # total RAM should be detectable on any normal dev machine
    assert hw.total_ram_bytes is None or hw.total_ram_bytes > 0
    # round-trips to a dict for --json
    assert hw.to_dict()["system"] == hw.system


def test_can_run_mlx_logic():
    apple = HardwareInfo(
        system="Darwin", machine="arm64", is_apple_silicon=True, chip="Apple M3",
        os_version="macOS 15.5", cpu_cores=12, performance_cores=8, efficiency_cores=4,
        gpu_cores=18, total_ram_bytes=36 * 1024 ** 3, mlx_available=True,
        recommended_working_set_bytes=27 * 1024 ** 3,
    )
    assert apple.can_run_mlx is True

    intel_mac = HardwareInfo(
        system="Darwin", machine="x86_64", is_apple_silicon=False, chip="Intel",
        os_version="macOS 13", cpu_cores=8, performance_cores=None, efficiency_cores=None,
        gpu_cores=None, total_ram_bytes=16 * 1024 ** 3, mlx_available=False,
        recommended_working_set_bytes=None,
    )
    assert intel_mac.can_run_mlx is False


def test_recommended_working_set_heuristic(monkeypatch):
    # With no metal device info, falls back to ~75% of total RAM.
    monkeypatch.setattr(hardware, "_query_metal_device_info", lambda: None)
    wss = hardware._detect_recommended_working_set(None, 100)
    assert wss == 75


def test_detect_gpu_cores_parsing():
    assert hardware._detect_gpu_cores(None) is None
    assert hardware._detect_gpu_cores({"gpu_core_count": 18}) == 18
    assert hardware._detect_gpu_cores({"num_gpu_cores": "30"}) == 30
    assert hardware._detect_gpu_cores({"unrelated": 1}) is None


def test_sysctl_missing_key_is_none():
    # A clearly bogus key should yield None rather than raising.
    assert hardware._sysctl("this.key.does.not.exist.zzz") is None
    assert hardware._sysctl_int("this.key.does.not.exist.zzz") is None
