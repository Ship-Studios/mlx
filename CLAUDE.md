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
mlx-runner serve -m <repo|path> --port 8080   # OpenAI-compatible HTTP server
```

`conftest.py` injects `src/` onto `sys.path`, so `pytest` works without an editable install.

## Architecture

The central design constraint: **the package must import cleanly on any platform**, even though `mlx`/`mlx-lm` only exist on Apple silicon. This shapes everything.

- **`hardware.py`** — `detect_hardware() -> HardwareInfo`. Best-effort probing via `sysctl` and `platform`; every field degrades to `None` rather than raising on non-macOS. `HardwareInfo.can_run_mlx` is the gate for whether generation can actually run here.
- **`memory.py`** — Pure-Python, dependency-free memory math (no mlx, no network). Estimates weight/KV-cache/overhead bytes and recommends a quantization level. Because it imports nothing heavy, it is the most thoroughly unit-tested module and runs anywhere.
- **`runner.py`** — `ModelRunner` wraps `mlx_lm.load` / `stream_generate`. **`mlx_lm` and `mlx.core` are imported lazily inside methods** (`_import_mlx_lm()`), never at module top level. Missing mlx-lm raises `MLXNotAvailableError`, not `ImportError`. `GenerationConfig`/`GenerationStats` are plain dataclasses. `stream()` is the primitive; `generate()` just joins it.
- **`cli.py`** — argparse with subparsers. **`info` and `fit` must work without mlx-lm installed**, so they only touch `hardware.py`/`memory.py`; `GenerationConfig` and `ModelRunner` are imported lazily inside the `generate`/`chat`/`cache` handlers. Each subparser sets `func` via `set_defaults`; `main()` dispatches through it. `serve` does **not** import mlx-lm in-process — it shells out to `python -m mlx_lm.server` via `subprocess` so the long-lived server runs in its own process; extra args after `--` are forwarded verbatim.

### Prompt caching (`cache` / `--prompt-cache-file`)

`ModelRunner` exposes `make_prompt_cache` / `load_prompt_cache` / `save_prompt_cache` (thin wraps over `mlx_lm.models.cache`, imported lazily via `_import_cache_module()`), plus `build_and_save_prompt_cache(context, path)` which streams the context through the model once to populate the cache, then saves it. `stream()`/`generate()` accept a `prompt_cache=` object and only add it to the `stream_generate` kwargs when non-`None`.

### Conventions to preserve when editing

- **Strict one-method-per-file rule**: Each file must contain exactly one class or function definition. No multiple methods in a single file.
- Never add a top-level `import mlx` or `import mlx_lm` anywhere in the package — keep them lazy and inside functions. The same applies to the CLI: don't import `runner` at module scope.
- Sizes are bytes everywhere internally; format only at the display boundary with `format_bytes` (binary/GiB) and the `human()` methods on the dataclasses.
- Dataclasses that cross the `--json` boundary expose `to_dict()`; keep it in sync with the dataclass fields.
- CLI exit codes are meaningful: `2` = bad args/usage, `3` = mlx-lm unavailable, and `fit` returns `1` when the model does not fit.

## Notes

- `README.md` is the user-facing CLI reference; this file (`CLAUDE.md`) covers internal architecture and conventions. Keep CLI flags/subcommands in sync across both when they change.
