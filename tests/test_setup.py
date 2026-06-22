import json

import pytest

from mlx_runner import catalog, cli, config, doctor
from mlx_runner.hardware import HardwareInfo

GIB = 1024 ** 3


def _hw(**overrides):
    base = dict(
        system="Darwin",
        machine="arm64",
        is_apple_silicon=True,
        chip="Apple M-test",
        os_version="macOS 15.0",
        cpu_cores=10,
        performance_cores=8,
        efficiency_cores=2,
        gpu_cores=16,
        total_ram_bytes=32 * GIB,
        mlx_available=True,
        recommended_working_set_bytes=24 * GIB,
    )
    base.update(overrides)
    return HardwareInfo(**base)


# --- catalog -----------------------------------------------------------------


def test_recommend_picks_largest_that_fits():
    # Huge budget -> the largest model in the catalog.
    rec = catalog.recommend_model(1024 * GIB, safety_fraction=0.9)
    assert rec is catalog.CATALOG[-1]


def test_recommend_small_budget_picks_small_model():
    rec = catalog.recommend_model(6 * GIB, safety_fraction=0.9)
    assert rec is not None
    # Should be one of the smaller builds, not the 72B.
    assert rec.params < 10_000_000_000


def test_recommend_returns_none_when_nothing_fits():
    assert catalog.recommend_model(1, safety_fraction=0.9) is None


def test_fitting_models_sorted_ascending():
    models = catalog.fitting_models(1024 * GIB)
    params = [m.params for m in models]
    assert params == sorted(params)
    assert len(models) == len(catalog.CATALOG)


# --- doctor ------------------------------------------------------------------


def test_doctor_apple_silicon_ok():
    checks = doctor.run_checks(_hw(is_apple_silicon=True))
    apple = next(c for c in checks if c.name == "Apple silicon")
    assert apple.status == doctor.OK


def test_doctor_intel_mac_fails():
    checks = doctor.run_checks(_hw(is_apple_silicon=False, system="Darwin", machine="x86_64"))
    apple = next(c for c in checks if c.name == "Apple silicon")
    assert apple.status == doctor.FAIL
    assert apple.remediation


def test_doctor_low_ram_warns():
    checks = doctor.run_checks(_hw(total_ram_bytes=4 * GIB))
    mem = next(c for c in checks if c.name == "Memory")
    assert mem.status == doctor.WARN


def test_worst_status_and_is_ready():
    ok = [doctor.Check("a", doctor.OK, "")]
    warn = ok + [doctor.Check("b", doctor.WARN, "")]
    fail = warn + [doctor.Check("c", doctor.FAIL, "")]
    assert doctor.worst_status(ok) == doctor.OK
    assert doctor.worst_status(warn) == doctor.WARN
    assert doctor.worst_status(fail) == doctor.FAIL
    assert doctor.is_ready(warn) is True
    assert doctor.is_ready(fail) is False


# --- cli integration ---------------------------------------------------------


def test_cli_doctor_json(capsys):
    rc = cli.main(["doctor", "--json"])
    data = json.loads(capsys.readouterr().out)
    assert isinstance(data, list) and data
    assert {"name", "status", "detail"} <= set(data[0])
    assert rc in (0, 1)


def test_cli_download_requires_a_model(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("MLX_RUNNER_CONFIG", str(tmp_path / "config.json"))
    rc = cli.main(["download"])  # no arg, no configured default
    assert rc == 2
    assert "no model" in capsys.readouterr().err.lower()


def test_cli_setup_force_sets_default_without_download(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("MLX_RUNNER_CONFIG", str(tmp_path / "config.json"))
    rc = cli.main(
        ["setup", "-m", "some/model", "--no-download", "--no-smoke-test", "--force"]
    )
    assert rc == 0
    assert config.load_config().model == "some/model"


def test_cli_setup_recommends_when_no_model(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("MLX_RUNNER_CONFIG", str(tmp_path / "config.json"))
    # Force past readiness; recommendation uses real host memory.
    rc = cli.main(["setup", "--no-download", "--no-smoke-test", "--force"])
    out = capsys.readouterr().out
    # Either it recommended something (and saved it) or reported nothing fits.
    if rc == 0:
        assert "Recommended:" in out
        assert config.load_config().model
    else:
        assert rc in (1, 2)
