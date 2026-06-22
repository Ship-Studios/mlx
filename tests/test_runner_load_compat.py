"""Regression tests for ModelRunner.load across mlx-lm versions.

`trust_remote_code` only became a top-level `mlx_lm.load()` argument in newer
mlx-lm; older versions reject it with TypeError. These tests fake `mlx_lm` with
each signature shape and assert ModelRunner.load adapts (no unsupported kwarg).
"""
import sys
import types

from mlx_runner.runner import ModelRunner


def _fake_mlx_lm(monkeypatch, load_fn):
    mod = types.ModuleType("mlx_lm")
    mod.load = load_fn
    monkeypatch.setitem(sys.modules, "mlx_lm", mod)


def test_old_mlx_lm_without_trust_remote_code_param(monkeypatch):
    # Older mlx-lm: explicit signature, NO trust_remote_code, NO **kwargs.
    captured = {}

    def load(path, tokenizer_config=None, model_config=None, adapter_path=None, lazy=False):
        captured["path"] = path
        captured["tokenizer_config"] = tokenizer_config
        captured["adapter_path"] = adapter_path
        return ("MODEL", "TOK")

    _fake_mlx_lm(monkeypatch, load)
    # This used to raise TypeError: load() got an unexpected keyword argument
    # 'trust_remote_code'. Now it must succeed by threading it via tokenizer_config.
    ModelRunner.load("repo", trust_remote_code=True)
    assert captured["path"] == "repo"
    assert captured["tokenizer_config"] == {"trust_remote_code": True}


def test_old_mlx_lm_trust_remote_code_false_omits_it(monkeypatch):
    captured = {}

    def load(path, tokenizer_config=None, model_config=None, adapter_path=None, lazy=False):
        captured["tokenizer_config"] = tokenizer_config
        return ("MODEL", "TOK")

    _fake_mlx_lm(monkeypatch, load)
    ModelRunner.load("repo")  # trust_remote_code defaults False
    assert captured["tokenizer_config"] == {}  # not threaded, not passed top-level


def test_new_mlx_lm_with_trust_remote_code_param(monkeypatch):
    captured = {}

    def load(path, tokenizer_config=None, model_config=None, adapter_path=None,
             lazy=False, trust_remote_code=False):
        captured["trust_remote_code"] = trust_remote_code
        captured["tokenizer_config"] = tokenizer_config
        return ("MODEL", "TOK")

    _fake_mlx_lm(monkeypatch, load)
    ModelRunner.load("repo", trust_remote_code=True)
    assert captured["trust_remote_code"] is True
    assert captured["tokenizer_config"] == {}  # passed top-level, not duplicated


def test_wrapped_load_advertises_kwargs_but_rejects_arg(monkeypatch):
    # A decorated/wrapped load whose signature shows **kwargs yet still rejects
    # trust_remote_code — signature inspection would be fooled; the try/except
    # retry must still recover by threading via tokenizer_config.
    captured = {}

    def load(path, tokenizer_config=None, adapter_path=None, lazy=False, **kwargs):
        if "trust_remote_code" in kwargs:
            raise TypeError("load() got an unexpected keyword argument 'trust_remote_code'")
        captured["tokenizer_config"] = tokenizer_config
        return ("MODEL", "TOK")

    _fake_mlx_lm(monkeypatch, load)
    ModelRunner.load("repo", trust_remote_code=True)  # must NOT raise
    assert captured["tokenizer_config"] == {"trust_remote_code": True}
