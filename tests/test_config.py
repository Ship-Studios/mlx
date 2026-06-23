import json

import pytest

from mlx_runner import cli, config


def test_defaults_when_no_file(tmp_path):
    cfg = config.load_config(tmp_path / "nope.json")
    assert cfg.model is None
    assert cfg.max_tokens == 512
    assert cfg.temperature == 0.0
    assert cfg.safety_fraction == 0.8


def test_save_then_load_roundtrip(tmp_path):
    p = tmp_path / "config.json"
    cfg = config.UserConfig(model="mlx-community/foo", temperature=0.7, max_tokens=256)
    written = config.save_config(cfg, p)
    assert written == p
    loaded = config.load_config(p)
    assert loaded.model == "mlx-community/foo"
    assert loaded.temperature == 0.7
    assert loaded.max_tokens == 256


def test_load_ignores_unknown_keys_and_bad_values(tmp_path):
    p = tmp_path / "config.json"
    p.write_text(json.dumps({"model": "x", "bogus": 1, "max_tokens": "not-int"}))
    cfg = config.load_config(p)
    assert cfg.model == "x"
    assert cfg.max_tokens == 512  # bad value falls back to default
    assert not hasattr(cfg, "bogus")


def test_load_malformed_json_is_default(tmp_path):
    p = tmp_path / "config.json"
    p.write_text("{not valid json")
    cfg = config.load_config(p)
    assert cfg == config.UserConfig()


def test_coerce_value_types():
    assert config.coerce_value("max_tokens", "128") == 128
    assert config.coerce_value("temperature", "0.8") == pytest.approx(0.8)
    assert config.coerce_value("model", "repo/id") == "repo/id"
    # optional fields clear to None on null-ish tokens
    assert config.coerce_value("repetition_penalty", "none") is None
    assert config.coerce_value("seed", "default") is None


def test_coerce_value_errors():
    with pytest.raises(KeyError):
        config.coerce_value("nope", "1")
    with pytest.raises(ValueError):
        config.coerce_value("max_tokens", "abc")


def test_default_config_path_respects_env(monkeypatch, tmp_path):
    target = tmp_path / "custom.json"
    monkeypatch.setenv("MLX_RUNNER_CONFIG", str(target))
    assert config.default_config_path() == target


def test_default_config_path_xdg(monkeypatch, tmp_path):
    monkeypatch.delenv("MLX_RUNNER_CONFIG", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    assert config.default_config_path() == tmp_path / "mlx-runner" / "config.json"


# --- CLI integration ---------------------------------------------------------


def test_cli_config_set_get_show(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("MLX_RUNNER_CONFIG", str(tmp_path / "config.json"))

    assert cli.main(["config", "set", "model", "mlx-community/foo"]) == 0
    capsys.readouterr()

    assert cli.main(["config", "get", "model"]) == 0
    assert capsys.readouterr().out.strip() == "mlx-community/foo"

    assert cli.main(["config", "show", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["model"] == "mlx-community/foo"


def test_cli_config_unset(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("MLX_RUNNER_CONFIG", str(tmp_path / "config.json"))
    cli.main(["config", "set", "temperature", "0.9"])
    capsys.readouterr()
    cli.main(["config", "unset", "temperature"])
    capsys.readouterr()
    assert config.load_config().temperature == 0.0


def test_configured_model_makes_flag_optional(monkeypatch, tmp_path):
    """With a default model set, `generate` should not require -m."""
    monkeypatch.setenv("MLX_RUNNER_CONFIG", str(tmp_path / "config.json"))
    cli.main(["config", "set", "model", "mlx-community/foo"])
    parser = cli.build_parser()
    args = parser.parse_args(["generate", "-p", "hi"])
    assert args.model == "mlx-community/foo"


def test_no_configured_model_still_requires_flag(monkeypatch, tmp_path):
    monkeypatch.setenv("MLX_RUNNER_CONFIG", str(tmp_path / "config.json"))
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["generate", "-p", "hi"])
