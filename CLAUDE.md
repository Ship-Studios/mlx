# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`mlx-runner` is a hardware-aware CLI for running and chatting with local LLMs on Apple silicon via [`mlx-lm`](https://github.com/ml-explore/mlx-lm). The package uses a `src/` layout (`src/mlx_runner/`) and is invoked as `mlx-runner` (console script) or `python -m mlx_runner`.

## Commands

```bash
pip install -e ".[dev]"   # install with pytest (mlx-lm is Darwin-only; see below)
pytest                    # run all tests (pythonpath/testpaths configured in pyproject.toml)
pytest tests/test_memory.py::test_name   # run a single test

mlx-runner info                          # detected hardware + MLX availability
mlx-runner fit 7B --bits 4               # does a 7B model fit at 4-bit on this machine?
mlx-runner generate -m <repo|path> -p "..."   # alias: run
mlx-runner chat -m <repo|path>           # interactive REPL
mlx-runner cache -m <repo|path> -c "long context" -o ctx.safetensors   # pre-compute a KV cache
mlx-runner generate -m <repo|path> --prompt-cache-file ctx.safetensors -p "Summarize"
mlx-runner serve -m <repo|path> --port 8080   # OpenAI-compatible server (shells out to mlx_lm.server)
mlx-runner serve -m <repo|path> --api anthropic --tunnel  # Anthropic Messages API, exposed via a Cloudflare tunnel
```

`conftest.py` injects `src/` onto `sys.path`, so `pytest` works without an editable install.

## Architecture

The central design constraint: **the package must import cleanly on any platform**, even though `mlx`/`mlx-lm` only exist on Apple silicon. This shapes everything.

**Every module is a subpackage** (the one-method-per-file rule, below): each logical module lives in `src/mlx_runner/<name>/`, whose files each hold exactly one function or one class. The package `__init__.py` re-exports the original public names, so import paths are unchanged (`from mlx_runner.runner import ModelRunner`, `from mlx_runner.memory import check_fit`, `from mlx_runner import detect_hardware`). Classes with 2+ methods are assembled from one single-method mixin per method; module-level constants/data live in `<name>/_constants.py` (a no-def file).

- **`hardware/`** — `detect_hardware() -> HardwareInfo`. Best-effort probing via `sysctl` and `platform`; every field degrades to `None` rather than raising on non-macOS. `HardwareInfo` (mixin-assembled from `can_run_mlx`/`to_dict`) and `HardwareInfo.can_run_mlx` gate whether generation can run here. `detect_hardware` calls its sibling probes through the package object (`import mlx_runner.hardware as _pkg; _pkg._query_metal_device_info()`) so tests can monkeypatch them.
- **`memory/`** — Pure-Python, dependency-free memory math (no mlx, no network). Estimates weight/KV-cache/overhead bytes and recommends a quantization level. Because it imports nothing heavy, it is the most thoroughly unit-tested code and runs anywhere. `MemoryEstimate`/`FitResult` are mixin-assembled dataclasses.
- **`runner/`** — `ModelRunner` (assembled from one mixin per method, in `_model_runner_*.py`) wraps `mlx_lm.load` / `stream_generate`. **`mlx_lm` and `mlx.core` are imported lazily inside the methods** (`_import_mlx_lm()` / `_import_cache_module()`), never at module top level. Missing mlx-lm raises `MLXNotAvailableError`, not `ImportError`. `load()` calls `_apply_memory_guard()` first — it lowers `mlx.core.set_memory_limit` to a fraction of the device's recommended working set (MLX defaults to **1.5x** it), so an oversized load raises a catchable OOM instead of panicking Apple's GPU driver (`IOGPUFamily` kernel panic) and taking the whole box down; it's best-effort and a no-op off-device. `GenerationConfig`/`GenerationStats` are dataclasses. `stream()` is the primitive; `generate()` just joins it.
- **`cli/`** — argparse with subparsers, one function per file. **`info` and `fit` must work without mlx-lm installed**, so they only touch `hardware`/`memory`; `GenerationConfig` and `ModelRunner` are imported lazily inside the `generate`/`chat`/`cache` handlers. `build_parser.py` wires each subparser's `func` via `set_defaults`; `main.py` dispatches through it. `cli/__init__.py` keeps a top-level `import subprocess` (and `doctor/__init__.py` keeps `import shutil`) so the `cli.subprocess`/`doctor.shutil` monkeypatch targets resolve. `serve` does **not** import mlx-lm in-process — the OpenAI path shells out to `python -m mlx_lm.server` via `subprocess`; extra args after `--` are forwarded verbatim. With no `--model` (and no configured default), `cmd_serve` calls `_select_serve_model(hw, safety_fraction=args.safety)` — it reuses `catalog.fitting_models` to offer only models the detected memory can hold (TTY: interactive pick; headless: auto-takes the largest fit), so a public/cloud `serve` can't blindly load an oversized model and crash the box via unified-memory exhaustion. `serve --safety` and the configured default for `fit`/`setup` are both `0.8` (lowered from `0.9`, which over-committed memory). The catalog fit estimate also reserves ~25% overhead (`CatalogModel.estimate`) since catalog entries carry no arch dims for an exact KV term; `setup`'s smoke-test load is additionally backstopped by the runner's `_apply_memory_guard`.
- **`anthropic_server/`** — in-process Anthropic Messages API emulator for `serve --api anthropic`. `make_handler(runner, *, api_key=None)` builds the HTTP handler around a **duck-typed `runner`** (only `.stream`/`.tokenizer`/`.last_stats` are used), so it needs no mlx; its nested handler class stays inside `make_handler.py`. `serve` imports `ModelRunner` lazily.
- Other subpackages: **`config/`** (persisted user defaults), **`doctor/`** (environment checks), **`catalog/`** (suggested-model table + fit), **`embeddings/`** (mlx-embeddings wrapper).

### Prompt caching (`cache` / `--prompt-cache-file`)

`ModelRunner` exposes `make_prompt_cache` / `load_prompt_cache` / `save_prompt_cache` (each its own mixin in `runner/_model_runner_*.py`; thin wraps over `mlx_lm.models.cache`, imported lazily via `_import_cache_module()`), plus `build_and_save_prompt_cache(context, path)` which streams the context through the model once to populate the cache, then saves it. `stream()`/`generate()` accept a `prompt_cache=` object and only add it to the `stream_generate` kwargs when non-`None`.

### Conventions to preserve when editing

- **Strict one-method-per-file rule**: every file holds at most one method/function. Each module is a subpackage `<name>/` of one-def files (`foo.py` = `def foo`; `<snake_class>.py` = the class). A class with 2+ methods is assembled from single-method mixins: each method is `_<snake_class>_<method>.py` defining `class _<Pascal><Method>Mixin`, and `<snake_class>.py` defines the real class inheriting all mixins in source order (`@dataclass` keeps its fields there; a plain assembly body is `pass`). Module-level constants go in `_constants.py`; the package `__init__.py` only re-exports (zero defs). Nested defs (a helper inside a function, the `Handler` class inside `make_handler`) stay nested and don't get their own file. Verify with an AST check: each non-`__init__`/`_constants`/`__main__` file has exactly one top-level def/class, and a lone class has ≤1 method.
- Never add a top-level `import mlx` / `import mlx_lm` / `import mlx_embeddings` anywhere in the package — keep them lazy, inside the functions/methods that need them (tests swap them via `sys.modules`). The CLI must not import `runner`/`anthropic_server` at module scope — only inside the handler that uses them.
- When splitting a function that a test monkeypatches by package attribute (e.g. `hardware._query_metal_device_info`, `cli.subprocess`, `doctor.shutil`), preserve the patch point: call it through the package/module object at runtime, and keep the relevant `import` in that package's `__init__.py`.
- Sizes are bytes everywhere internally; format only at the display boundary with `format_bytes` (binary/GiB) and the `human()` methods on the dataclasses.
- Dataclasses that cross the `--json` boundary expose `to_dict()`; keep it in sync with the dataclass fields.
- CLI exit codes are meaningful: `2` = bad args/usage, `3` = mlx-lm unavailable, and `fit` returns `1` when the model does not fit.

## Notes

- `README.md` is the user-facing CLI reference; this file (`CLAUDE.md`) covers internal architecture and conventions. Keep CLI flags/subcommands in sync across both when they change.
