"""Tests for ModelRunner that fake the mlx-lm / mlx modules in sys.modules.

This lets us exercise the orchestration (prompt formatting, sampler kwargs,
streaming, stats capture) without an Apple-silicon machine or a real model.
"""
import sys
import types

import pytest

from mlx_runner.runner import (
    GenerationConfig,
    GenerationStats,
    MLXNotAvailableError,
    ModelRunner,
)


class FakeResponse:
    def __init__(self, text, **stats):
        self.text = text
        self.prompt_tokens = stats.get("prompt_tokens", 0)
        self.generation_tokens = stats.get("generation_tokens", 0)
        self.prompt_tps = stats.get("prompt_tps", 0.0)
        self.generation_tps = stats.get("generation_tps", 0.0)
        self.peak_memory = stats.get("peak_memory", 0.0)
        self.finish_reason = stats.get("finish_reason", None)


class FakeTokenizer:
    def __init__(self, chat_template="{{ tmpl }}"):
        self.chat_template = chat_template
        self.captured_messages = None

    def apply_chat_template(self, messages, add_generation_prompt=True):
        self.captured_messages = list(messages)
        # mimic returning token ids
        return [1, 2, 3, len(messages)]


@pytest.fixture
def fake_mlx(monkeypatch):
    """Install fake mlx_lm, mlx_lm.sample_utils, mlx, mlx.core modules."""
    calls = {"stream_kwargs": None, "load_args": None, "seed": None,
             "sampler_args": None, "logits_args": None}

    mlx_lm = types.ModuleType("mlx_lm")

    def stream_generate(model, tokenizer, prompt, **kwargs):
        calls["stream_kwargs"] = kwargs
        calls["stream_prompt"] = prompt
        yield FakeResponse("Hello")
        yield FakeResponse(
            " world",
            prompt_tokens=5,
            generation_tokens=2,
            prompt_tps=100.0,
            generation_tps=40.0,
            peak_memory=3.5,
            finish_reason="stop",
        )

    def load(path, **kwargs):
        calls["load_args"] = (path, kwargs)
        return ("FAKE_MODEL", FakeTokenizer())

    mlx_lm.stream_generate = stream_generate
    mlx_lm.load = load

    sample_utils = types.ModuleType("mlx_lm.sample_utils")

    def make_sampler(**kwargs):
        calls["sampler_args"] = kwargs
        return "SAMPLER"

    def make_logits_processors(**kwargs):
        calls["logits_args"] = kwargs
        return ["PROC"]

    sample_utils.make_sampler = make_sampler
    sample_utils.make_logits_processors = make_logits_processors

    mlx = types.ModuleType("mlx")
    mlx_core = types.ModuleType("mlx.core")
    random_mod = types.SimpleNamespace(seed=lambda s: calls.__setitem__("seed", s))
    mlx_core.random = random_mod
    mlx.core = mlx_core

    monkeypatch.setitem(sys.modules, "mlx_lm", mlx_lm)
    monkeypatch.setitem(sys.modules, "mlx_lm.sample_utils", sample_utils)
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", mlx_core)
    return calls


def test_load_passes_through_args(fake_mlx):
    runner = ModelRunner.load("some/repo", adapter_path="adp", trust_remote_code=True)
    assert runner.model == "FAKE_MODEL"
    assert isinstance(runner.tokenizer, FakeTokenizer)
    path, kwargs = fake_mlx["load_args"]
    assert path == "some/repo"
    assert kwargs["adapter_path"] == "adp"
    assert kwargs["trust_remote_code"] is True
    assert kwargs["tokenizer_config"] == {}


def test_format_prompt_builds_messages():
    tok = FakeTokenizer()
    runner = ModelRunner("m", tok)
    runner.format_prompt(prompt="hi", system="be nice")
    assert tok.captured_messages == [
        {"role": "system", "content": "be nice"},
        {"role": "user", "content": "hi"},
    ]


def test_format_prompt_requires_input():
    runner = ModelRunner("m", FakeTokenizer())
    with pytest.raises(ValueError):
        runner.format_prompt()


def test_format_prompt_fallback_without_template():
    runner = ModelRunner("m", FakeTokenizer(chat_template=None))
    out = runner.format_prompt(
        messages=[{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
    )
    assert out == "a\nb"


def test_stream_yields_deltas_and_captures_stats(fake_mlx):
    runner = ModelRunner("m", FakeTokenizer())
    deltas = list(runner.stream(prompt="hi", config=GenerationConfig(max_tokens=10)))
    assert deltas == ["Hello", " world"]
    assert isinstance(runner.last_stats, GenerationStats)
    assert runner.last_stats.prompt_tokens == 5
    assert runner.last_stats.generation_tokens == 2
    assert runner.last_stats.generation_tps == 40.0
    assert runner.last_stats.peak_memory_gb == 3.5
    assert runner.last_stats.finish_reason == "stop"


def test_generate_concatenates(fake_mlx):
    runner = ModelRunner("m", FakeTokenizer())
    text = runner.generate(prompt="hi")
    assert text == "Hello world"


def test_build_kwargs_maps_config(fake_mlx):
    runner = ModelRunner("m", FakeTokenizer())
    cfg = GenerationConfig(
        max_tokens=33,
        temperature=0.7,
        top_p=0.9,
        top_k=40,
        min_p=0.05,
        repetition_penalty=1.1,
        max_kv_size=2048,
        kv_bits=8,
    )
    list(runner.stream(prompt="hi", config=cfg))
    kwargs = fake_mlx["stream_kwargs"]
    assert kwargs["max_tokens"] == 33
    assert kwargs["sampler"] == "SAMPLER"
    assert kwargs["logits_processors"] == ["PROC"]
    assert kwargs["max_kv_size"] == 2048
    assert kwargs["kv_bits"] == 8
    # sampler/processor builders got the right values
    assert fake_mlx["sampler_args"]["temp"] == 0.7
    assert fake_mlx["sampler_args"]["top_p"] == 0.9
    assert fake_mlx["sampler_args"]["top_k"] == 40
    assert fake_mlx["logits_args"]["repetition_penalty"] == 1.1


def test_build_kwargs_omits_optional_when_unset(fake_mlx):
    runner = ModelRunner("m", FakeTokenizer())
    list(runner.stream(prompt="hi", config=GenerationConfig()))
    kwargs = fake_mlx["stream_kwargs"]
    assert "max_kv_size" not in kwargs
    assert "kv_bits" not in kwargs


def test_seed_is_applied(fake_mlx):
    runner = ModelRunner("m", FakeTokenizer())
    list(runner.stream(prompt="hi", config=GenerationConfig(seed=123)))
    assert fake_mlx["seed"] == 123


def test_import_error_becomes_mlx_not_available(monkeypatch):
    # Ensure importing mlx_lm fails.
    monkeypatch.setitem(sys.modules, "mlx_lm", None)
    with pytest.raises(MLXNotAvailableError):
        ModelRunner.load("repo")


def test_stream_threads_prompt_cache(fake_mlx):
    runner = ModelRunner("m", FakeTokenizer())
    list(runner.stream(prompt="hi", prompt_cache="CACHE"))
    assert fake_mlx["stream_kwargs"]["prompt_cache"] == "CACHE"


def test_stream_omits_prompt_cache_when_none(fake_mlx):
    runner = ModelRunner("m", FakeTokenizer())
    list(runner.stream(prompt="hi"))
    assert "prompt_cache" not in fake_mlx["stream_kwargs"]


@pytest.fixture
def fake_cache_module(monkeypatch):
    """Install a fake mlx_lm.models.cache module."""
    calls = {}
    cache_mod = types.ModuleType("mlx_lm.models.cache")

    def make_prompt_cache(model, max_kv_size=None):
        calls["make"] = {"model": model, "max_kv_size": max_kv_size}
        return ["CACHE_STATE"]

    def save_prompt_cache(path, cache):
        calls["save"] = {"path": path, "cache": cache}

    def load_prompt_cache(path):
        calls["load"] = {"path": path}
        return ["LOADED_CACHE"]

    cache_mod.make_prompt_cache = make_prompt_cache
    cache_mod.save_prompt_cache = save_prompt_cache
    cache_mod.load_prompt_cache = load_prompt_cache

    # _import_cache_module imports mlx_lm first; a bare stub satisfies it, but
    # don't clobber a richer mlx_lm already installed by the fake_mlx fixture.
    if "mlx_lm" not in sys.modules:
        monkeypatch.setitem(sys.modules, "mlx_lm", types.ModuleType("mlx_lm"))
    monkeypatch.setitem(sys.modules, "mlx_lm.models", types.ModuleType("mlx_lm.models"))
    monkeypatch.setitem(sys.modules, "mlx_lm.models.cache", cache_mod)
    return calls


def test_cache_helpers(fake_cache_module):
    runner = ModelRunner("MODEL", FakeTokenizer())
    cache = runner.make_prompt_cache(max_kv_size=2048)
    assert cache == ["CACHE_STATE"]
    assert fake_cache_module["make"] == {"model": "MODEL", "max_kv_size": 2048}

    runner.save_prompt_cache("out.safetensors", cache)
    assert fake_cache_module["save"]["path"] == "out.safetensors"

    loaded = runner.load_prompt_cache("in.safetensors")
    assert loaded == ["LOADED_CACHE"]
    assert fake_cache_module["load"]["path"] == "in.safetensors"


def test_build_and_save_prompt_cache(fake_mlx, fake_cache_module):
    runner = ModelRunner("MODEL", FakeTokenizer())
    runner.build_and_save_prompt_cache("a long context", "doc.safetensors")
    # The context was streamed through the model with the freshly-made cache...
    assert fake_mlx["stream_kwargs"]["prompt_cache"] == ["CACHE_STATE"]
    # ...and the populated cache was saved to the requested path.
    assert fake_cache_module["save"]["path"] == "doc.safetensors"
