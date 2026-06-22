# mlx-runner

A hardware-aware CLI for running and chatting with local LLMs on Apple silicon via [`mlx-lm`](https://github.com/ml-explore/mlx-lm).

`mlx-runner` detects what your Mac can handle, tells you whether a given model will fit in memory (and at which quantization), and then loads and runs it — with streaming generation and an interactive chat REPL.

## Requirements

- **macOS on Apple silicon** (M1/M2/M3/…) to actually load and run models. `mlx-lm` pulls in `mlx`, which is Apple-silicon-only.
- **Python 3.8+**

The `info` and `fit` commands work on **any platform** (Linux, Intel Macs, etc.) — they only inspect hardware and do memory math, so you can plan before you're on the right machine.

## Installation

```bash
pip install -e .          # core install
pip install -e ".[dev]"   # plus pytest for development
```

This installs the `mlx-runner` console script. You can also invoke the package directly with `python -m mlx_runner`.

## Onboard a new Mac

**From a bare Mac**, `install.sh` bootstraps everything — it verifies the machine, finds (or `brew install`s) a recent Python, installs the `mlx-runner` package (via `pipx` when available), and then runs `mlx-runner setup`:

```bash
git clone <this-repo> && cd mlx && ./install.sh
# install only, skip onboarding:           ./install.sh --skip-setup
# pick your own model (forwarded to setup): ./install.sh -m mlx-community/Qwen2.5-7B-Instruct-4bit
```

(If you have no Python at all and no Homebrew, the script tells you to install [Homebrew](https://brew.sh) first — it won't run the Homebrew installer for you.)

**If the package is already installed**, one command gets the Mac ready to run LLMs:

```bash
mlx-runner setup
```

`setup` runs the readiness checks, recommends the largest catalog model that fits this machine's memory, downloads it, saves it as your default model, and runs a quick smoke-test generation. From then on you can just `mlx-runner chat` (no `-m` needed).

Useful flags: `--model/-m` to skip the recommendation and pick your own, `--no-download` / `--no-smoke-test` to do less, and `--force` to configure even when a readiness check fails. To inspect readiness on its own, use `mlx-runner doctor`.

## Usage

### `setup` — onboard this Mac (start here)

```bash
mlx-runner setup                      # recommend + download + set default + smoke test
mlx-runner setup -m mlx-community/Qwen2.5-7B-Instruct-4bit
mlx-runner setup --no-smoke-test      # skip the test generation
```

Exit code is `0` on success, `1` if no model fits or a step fails, and non-zero (with guidance) if the machine isn't ready and `--force` wasn't given.

### `doctor` — readiness check

```bash
mlx-runner doctor
mlx-runner doctor --json
```

Reports a pass/fail line for each prerequisite — Apple silicon, Python version, `mlx`/`mlx-lm` importability, and memory — with remediation hints. Exit code is `1` if any hard requirement fails.

### `download` — pre-fetch a model

```bash
mlx-runner download mlx-community/Qwen2.5-7B-Instruct-4bit
mlx-runner download           # uses the configured default model
```

Fetches the model snapshot into the Hugging Face cache so the first `generate`/`chat` isn't a cold download. Prints the local path.

### `info` — inspect your machine

```bash
mlx-runner info
mlx-runner info --json
```

Shows the chip, CPU/GPU core counts, total and recommended-usable RAM, whether `mlx` is importable, and whether mlx-lm can actually execute here.

### `fit` — will a model fit?

Estimate the memory footprint of a model and whether it fits within a safe fraction of available memory:

```bash
mlx-runner fit 7B               # full-precision (fp16)
mlx-runner fit 7B --bits 4      # 4-bit quantized
mlx-runner fit 1.5B --bits 8 --seq-len 8192
mlx-runner fit 13B --json
```

Parameter counts accept `K`/`M`/`B`/`T` suffixes (e.g. `350M`, `1.5B`) or a plain integer. The output reports the estimated breakdown (weights / KV-cache / overhead) and recommends the highest-quality quantization that fits. Pass KV-cache dimensions (`--layers`, `--kv-heads`, `--head-dim`) to include the cache in the estimate.

Exit code is `1` when the model does not fit, `0` when it does.

### `generate` (alias `run`) — one-shot generation

```bash
mlx-runner generate -m mlx-community/Qwen2.5-7B-Instruct-4bit -p "Explain unified memory."
echo "Write a haiku about Metal." | mlx-runner run -m <model>
mlx-runner generate -m <model> -p "..." --system "You are terse." --stats --no-stream
```

`--model` accepts a Hugging Face repo id or a local path. The prompt comes from `--prompt` or, if omitted, stdin. Tokens stream by default; `--no-stream` prints the full completion at once. `--stats` prints timing and peak-memory stats to stderr.

#### Suggested models

Pre-quantized 4-bit builds from the [`mlx-community`](https://huggingface.co/mlx-community) org, smallest to largest — use `mlx-runner fit` to check which suits your RAM:

| Model | Repo id |
| --- | --- |
| Llama 3.2 1B | `mlx-community/Llama-3.2-1B-Instruct-4bit` |
| Qwen2.5 1.5B | `mlx-community/Qwen2.5-1.5B-Instruct-4bit` |
| Llama 3.2 3B | `mlx-community/Llama-3.2-3B-Instruct-4bit` |
| Qwen2.5 7B | `mlx-community/Qwen2.5-7B-Instruct-4bit` |
| Llama 3.1 8B | `mlx-community/Meta-Llama-3.1-8B-Instruct-4bit` |
| Qwen2.5 14B | `mlx-community/Qwen2.5-14B-Instruct-4bit` |

### `chat` — interactive REPL

```bash
mlx-runner chat -m <model> --system "You are a helpful assistant."
```

Maintains conversation history across turns. Type `exit`/`quit` or press Ctrl-D to leave.

### `cache` — pre-compute a reusable prompt cache

For a long context you query repeatedly (a document, a codebase, a system prompt), pre-compute its KV activations once and reuse them so the context isn't re-encoded on every call:

```bash
# Build the cache from a file (or --context "...")
cat long_document.txt | mlx-runner cache -m <model> -o doc.safetensors

# Reuse it across many fast queries
mlx-runner generate -m <model> --prompt-cache-file doc.safetensors -p "Summarize the document."
mlx-runner generate -m <model> --prompt-cache-file doc.safetensors -p "List the action items."
```

The context comes from `--context`/`-c` or stdin. `--max-kv-size` caps the cache size.

### `serve` — OpenAI-compatible HTTP server

Launch [`mlx_lm.server`](https://github.com/ml-explore/mlx-lm), which exposes an OpenAI-compatible API (`POST /v1/chat/completions`):

```bash
mlx-runner serve -m <model> --host 127.0.0.1 --port 8080
# forward extra flags straight to mlx_lm.server after `--`:
mlx-runner serve -m <model> -- --log-level DEBUG
```

Any OpenAI client can then point at `http://127.0.0.1:8080/v1`. This runs `mlx_lm.server` in a subprocess, so it needs `mlx-lm` installed on an Apple-silicon Mac.

### `config` — persisted defaults

Save a default model and default sampling parameters so you don't have to repeat them on every call:

```bash
mlx-runner config set model mlx-community/Qwen2.5-7B-Instruct-4bit
mlx-runner config set temperature 0.7
mlx-runner config show           # or --json
mlx-runner config get model
mlx-runner config unset temperature   # back to the built-in default
mlx-runner config path           # where the file lives
```

Once a default model is set, `-m` becomes optional for `generate`, `chat`, `cache`, and `serve`. Any flag still wins over the saved value for that one invocation. Optional keys (e.g. `repetition_penalty`, `seed`) accept `none` to clear them.

The file is JSON and its location resolves in this order: `$MLX_RUNNER_CONFIG`, then `$XDG_CONFIG_HOME/mlx-runner/config.json`, then `~/.config/mlx-runner/config.json`. A missing or malformed file simply falls back to built-in defaults.

### Generation options

`generate` and `chat` share these sampling/decoding flags (defaults below are the built-in ones; a saved `config` overrides them, and an explicit flag overrides the config):

| Flag | Default | Notes |
| --- | --- | --- |
| `--max-tokens` | `512` | |
| `--temp` / `--temperature` | `0.0` | greedy by default |
| `--top-p` | `0.0` | |
| `--top-k` | `0` | |
| `--min-p` | `0.0` | |
| `--repetition-penalty` | none | |
| `--seed` | none | for reproducible sampling |
| `--max-kv-size` | none | cap the KV cache |
| `--kv-bits` | none | quantize the KV cache |
| `--adapter-path` | none | optional LoRA adapter |
| `--trust-remote-code` | off | |
| `--prompt-cache-file` | none | `generate` only — reuse a cache built by `cache` |

## Development

```bash
pytest                                   # run the full suite
pytest tests/test_memory.py::test_name   # run a single test
```

The package uses a `src/` layout; `conftest.py` puts `src/` on `sys.path`, so the tests run without an editable install. The hardware-detection and memory-estimation modules are pure-Python and dependency-free, so the bulk of the suite runs on any platform — `mlx`/`mlx-lm` are imported lazily and only needed to actually load a model.

## How it works

- **`mlx_runner.hardware`** — best-effort hardware detection via `sysctl`/`platform`, degrading gracefully off Apple silicon.
- **`mlx_runner.memory`** — pure-Python estimation of weight, KV-cache, and overhead memory, plus a quantization recommender.
- **`mlx_runner.runner`** — `ModelRunner`, a thin wrapper over `mlx_lm.load` / `stream_generate` with lazy mlx imports, plus prompt-cache helpers over `mlx_lm.models.cache`.
- **`mlx_runner.config`** — a small JSON store of user defaults (default model, sampling params) loaded into the CLI's argument defaults.
- **`mlx_runner.doctor`** — pure-Python readiness checks (Apple silicon, Python, mlx/mlx-lm, memory) with a severity for each.
- **`mlx_runner.catalog`** — a curated list of mlx-community models with approximate sizes, used to recommend the largest one that fits.
- **`mlx_runner.cli`** — the argparse front end; `info`/`fit`/`config` work without mlx-lm installed, and `serve` shells out to `mlx_lm.server`.

## License

MIT
