import sys
import types

import pytest

from mlx_runner import cli


def test_build_parser_requires_subcommand():
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_info_text(capsys):
    rc = cli.main(["info"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "System:" in out
    assert "MLX importable:" in out


def test_info_json(capsys):
    rc = cli.main(["info", "--json"])
    assert rc == 0
    import json

    data = json.loads(capsys.readouterr().out)
    assert "system" in data and "mlx_available" in data


def test_fit_returns_zero_when_fits(capsys):
    # tiny model in fp16 fits almost anywhere
    rc = cli.main(["fit", "100M", "--safety", "0.9"])
    out = capsys.readouterr().out
    assert "Model:" in out and "Memory:" in out
    assert rc in (0, 1)  # depends on host RAM, but must be a valid verdict


def test_fit_json(capsys):
    rc = cli.main(["fit", "7B", "--bits", "4", "--json"])
    import json

    data = json.loads(capsys.readouterr().out)
    assert data["num_params"] == 7_000_000_000
    assert "estimate" in data and "recommended" in data
    assert rc in (0, 1)


def test_fit_bad_params(capsys):
    rc = cli.main(["fit", "not-a-number"])
    assert rc == 2
    assert "error" in capsys.readouterr().err


def test_generate_uses_runner(monkeypatch, capsys):
    """generate should load a runner and stream its output to stdout."""
    captured = {}

    class FakeRunner:
        last_stats = None

        @classmethod
        def load(cls, model, **kwargs):
            captured["model"] = model
            captured["load_kwargs"] = kwargs
            return cls()

        def stream(self, prompt=None, system=None, config=None, prompt_cache=None):
            captured["prompt"] = prompt
            captured["system"] = system
            captured["config"] = config
            captured["prompt_cache"] = prompt_cache
            yield "hello "
            yield "there"

    fake_runner_mod = types.ModuleType("mlx_runner.runner")
    fake_runner_mod.ModelRunner = FakeRunner
    fake_runner_mod.MLXNotAvailableError = RuntimeError

    class FakeConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    fake_runner_mod.GenerationConfig = FakeConfig
    monkeypatch.setitem(sys.modules, "mlx_runner.runner", fake_runner_mod)

    rc = cli.main(["generate", "-m", "some/model", "-p", "hi"])
    assert rc == 0
    assert captured["model"] == "some/model"
    assert captured["prompt"] == "hi"
    out = capsys.readouterr().out
    assert "hello there" in out


def test_generate_no_prompt_no_stdin(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", types.SimpleNamespace(read=lambda: ""))
    rc = cli.main(["generate", "-m", "x"])
    assert rc == 2
    assert "no prompt" in capsys.readouterr().err.lower()


def test_generate_loads_prompt_cache(monkeypatch, capsys):
    """`--prompt-cache-file` should load a cache and thread it into stream()."""
    captured = {}

    class FakeRunner:
        last_stats = None

        @classmethod
        def load(cls, model, **kwargs):
            return cls()

        def load_prompt_cache(self, path):
            captured["cache_path"] = path
            return "CACHE_OBJ"

        def stream(self, prompt=None, system=None, config=None, prompt_cache=None):
            captured["prompt_cache"] = prompt_cache
            yield "ok"

    fake_runner_mod = types.ModuleType("mlx_runner.runner")
    fake_runner_mod.ModelRunner = FakeRunner
    fake_runner_mod.MLXNotAvailableError = RuntimeError
    fake_runner_mod.GenerationConfig = lambda **k: types.SimpleNamespace(**k)
    monkeypatch.setitem(sys.modules, "mlx_runner.runner", fake_runner_mod)

    rc = cli.main(
        ["generate", "-m", "m", "-p", "hi", "--prompt-cache-file", "ctx.safetensors"]
    )
    assert rc == 0
    assert captured["cache_path"] == "ctx.safetensors"
    assert captured["prompt_cache"] == "CACHE_OBJ"


def test_cache_builds_and_saves(monkeypatch, capsys):
    """`cache` should read context and call build_and_save_prompt_cache."""
    captured = {}

    class FakeRunner:
        @classmethod
        def load(cls, model, **kwargs):
            return cls()

        def build_and_save_prompt_cache(self, context, out, max_kv_size=None):
            captured["context"] = context
            captured["out"] = out
            captured["max_kv_size"] = max_kv_size

    fake_runner_mod = types.ModuleType("mlx_runner.runner")
    fake_runner_mod.ModelRunner = FakeRunner
    fake_runner_mod.MLXNotAvailableError = RuntimeError
    monkeypatch.setitem(sys.modules, "mlx_runner.runner", fake_runner_mod)

    rc = cli.main(
        ["cache", "-m", "m", "-c", "long context", "-o", "out.safetensors",
         "--max-kv-size", "4096"]
    )
    assert rc == 0
    assert captured == {
        "context": "long context",
        "out": "out.safetensors",
        "max_kv_size": 4096,
    }
    assert "Saved prompt cache" in capsys.readouterr().err


def test_cache_no_context(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", types.SimpleNamespace(read=lambda: "   "))
    rc = cli.main(["cache", "-m", "m", "-o", "out.safetensors"])
    assert rc == 2
    assert "no context" in capsys.readouterr().err.lower()


def test_serve_invokes_mlx_lm_server(monkeypatch, capsys):
    """`serve` should shell out to `python -m mlx_lm.server` with the right args."""
    captured = {}

    def fake_call(cmd):
        captured["cmd"] = cmd
        return 0

    monkeypatch.setattr(cli.subprocess, "call", fake_call)

    rc = cli.main(
        ["serve", "-m", "some/model", "--host", "0.0.0.0", "--port", "9000",
         "--", "--max-tokens", "100"]
    )
    assert rc == 0
    cmd = captured["cmd"]
    assert cmd[:3] == [sys.executable, "-m", "mlx_lm.server"]
    assert "--model" in cmd and "some/model" in cmd
    assert cmd[cmd.index("--host") + 1] == "0.0.0.0"
    assert cmd[cmd.index("--port") + 1] == "9000"
    # forwarded extra args, with the separating "--" stripped
    assert cmd[-2:] == ["--max-tokens", "100"]
    assert "--" not in cmd
