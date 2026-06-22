"""Command-line interface for mlx_runner.

Subcommands:
  info      Show detected hardware and MLX availability.
  fit       Estimate whether a model of N parameters fits in memory.
  generate  Generate text from a prompt (alias: run).
  chat      Interactive chat REPL.
  cache     Pre-compute and save a reusable prompt (KV) cache for a long context.
  serve     Serve an HTTP API — OpenAI-compatible (mlx_lm.server) or Anthropic Messages.
  config    View or change persisted defaults (default model, sampling params).
  doctor    Check whether this machine is ready to run LLMs.
  download  Pre-download a model from Hugging Face into the local cache.
  setup     Onboard this Mac: readiness check, pick + download a model, set default.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
from typing import List, Optional

from . import __version__
from .catalog import recommend_model
from .config import (
    UserConfig,
    coerce_value,
    default_config_path,
    known_keys,
    load_config,
    save_config,
)
from .doctor import FAIL, OK, WARN, Check, is_ready, run_checks
from .hardware import detect_hardware
from .memory import (
    check_fit,
    estimate_model_memory,
    format_bytes,
    parse_param_count,
    recommend_quantization,
)


def _add_generation_args(p: argparse.ArgumentParser, config: UserConfig) -> None:
    g = p.add_argument_group("generation")
    g.add_argument("--max-tokens", type=int, default=config.max_tokens)
    g.add_argument(
        "--temp", "--temperature", dest="temperature", type=float,
        default=config.temperature,
    )
    g.add_argument("--top-p", type=float, default=config.top_p)
    g.add_argument("--top-k", type=int, default=config.top_k)
    g.add_argument("--min-p", type=float, default=config.min_p)
    g.add_argument("--repetition-penalty", type=float, default=config.repetition_penalty)
    g.add_argument("--seed", type=int, default=config.seed)
    g.add_argument("--max-kv-size", type=int, default=config.max_kv_size)
    g.add_argument("--kv-bits", type=int, default=config.kv_bits)


def _config_from_args(args) -> "object":
    # Imported lazily so `info`/`fit` work without mlx-lm present.
    from .runner import GenerationConfig

    return GenerationConfig(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        repetition_penalty=args.repetition_penalty,
        seed=args.seed,
        max_kv_size=args.max_kv_size,
        kv_bits=args.kv_bits,
    )


def build_parser(config: Optional[UserConfig] = None) -> argparse.ArgumentParser:
    if config is None:
        config = load_config()
    parser = argparse.ArgumentParser(
        prog="mlx-runner",
        description="Hardware-aware local LLM runner built on mlx-lm.",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_info = sub.add_parser("info", help="Show detected hardware and MLX availability.")
    p_info.add_argument("--json", action="store_true", help="Emit JSON.")
    p_info.set_defaults(func=cmd_info)

    p_fit = sub.add_parser("fit", help="Estimate whether a model fits in memory.")
    p_fit.add_argument(
        "params", help="Parameter count, e.g. 7B, 13B, 1.5B, 350M, or 7000000000."
    )
    p_fit.add_argument(
        "--bits", type=int, default=None, help="Quantization bits (e.g. 4, 8). Omit for fp16."
    )
    p_fit.add_argument("--dtype", default="float16")
    p_fit.add_argument("--group-size", type=int, default=64)
    p_fit.add_argument("--seq-len", type=int, default=4096)
    p_fit.add_argument("--layers", type=int, default=None)
    p_fit.add_argument("--kv-heads", type=int, default=None)
    p_fit.add_argument("--head-dim", type=int, default=None)
    p_fit.add_argument(
        "--safety", type=float, default=config.safety_fraction,
        help="Fraction of memory usable (0-1].",
    )
    p_fit.add_argument("--json", action="store_true")
    p_fit.set_defaults(func=cmd_fit)

    p_gen = sub.add_parser(
        "generate", aliases=["run"], help="Generate text from a prompt."
    )
    p_gen.add_argument(
        "--model", "-m", default=config.model, required=config.model is None,
        help="HF repo id or local path." + _model_default_note(config),
    )
    p_gen.add_argument("--prompt", "-p", help="User prompt; reads stdin if omitted.")
    p_gen.add_argument("--system", default=config.system, help="Optional system prompt.")
    p_gen.add_argument("--adapter-path", default=None, help="Optional LoRA adapter path.")
    p_gen.add_argument("--no-stream", action="store_true", help="Disable token streaming.")
    p_gen.add_argument("--stats", action="store_true", help="Print stats to stderr.")
    p_gen.add_argument("--trust-remote-code", action="store_true")
    p_gen.add_argument(
        "--prompt-cache-file",
        default=None,
        help="Reuse a prompt cache saved by `cache` so the context isn't re-encoded.",
    )
    _add_generation_args(p_gen, config)
    p_gen.set_defaults(func=cmd_generate)

    p_chat = sub.add_parser("chat", help="Interactive chat REPL.")
    p_chat.add_argument(
        "--model", "-m", default=config.model, required=config.model is None,
        help="HF repo id or local path." + _model_default_note(config),
    )
    p_chat.add_argument("--system", default=config.system, help="Optional system prompt.")
    p_chat.add_argument("--adapter-path", default=None)
    p_chat.add_argument("--trust-remote-code", action="store_true")
    _add_generation_args(p_chat, config)
    p_chat.set_defaults(func=cmd_chat)

    p_cache = sub.add_parser(
        "cache", help="Pre-compute a reusable prompt (KV) cache for a long context."
    )
    p_cache.add_argument(
        "--model", "-m", default=config.model, required=config.model is None,
        help="HF repo id or local path." + _model_default_note(config),
    )
    p_cache.add_argument(
        "--context", "-c", help="Context text to cache; reads stdin if omitted."
    )
    p_cache.add_argument(
        "--out", "-o", required=True, help="Output path for the cache (.safetensors)."
    )
    p_cache.add_argument("--adapter-path", default=None)
    p_cache.add_argument("--trust-remote-code", action="store_true")
    p_cache.add_argument("--max-kv-size", type=int, default=None)
    p_cache.set_defaults(func=cmd_cache)

    p_serve = sub.add_parser(
        "serve",
        help="Serve an HTTP API (OpenAI-compatible via mlx_lm.server, or Anthropic Messages).",
    )
    p_serve.add_argument(
        "--model", "-m", default=config.model, help="Default model to serve."
    )
    p_serve.add_argument(
        "--api", choices=["openai", "anthropic"], default="openai",
        help="Wire format to serve (default: openai via mlx_lm.server).",
    )
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8080)
    p_serve.add_argument("--adapter-path", default=None)
    p_serve.add_argument("--trust-remote-code", action="store_true")
    p_serve.add_argument(
        "--api-key", default=None,
        help="anthropic: require this value in the x-api-key header (default: open).",
    )
    p_serve.add_argument(
        "--tunnel", action="store_true",
        help="Expose the server publicly via a Cloudflare quick tunnel (needs cloudflared).",
    )
    p_serve.add_argument(
        "server_args",
        nargs=argparse.REMAINDER,
        help="openai: extra args forwarded verbatim to mlx_lm.server (after --).",
    )
    p_serve.set_defaults(func=cmd_serve)

    p_config = sub.add_parser(
        "config", help="View or change persisted defaults (default model, sampling)."
    )
    csub = p_config.add_subparsers(dest="config_command", required=True)

    c_show = csub.add_parser("show", help="Print the current configuration.")
    c_show.add_argument("--json", action="store_true", help="Emit JSON.")
    c_show.set_defaults(func=cmd_config_show)

    c_path = csub.add_parser("path", help="Print the config file path.")
    c_path.set_defaults(func=cmd_config_path)

    c_get = csub.add_parser("get", help="Print a single configuration value.")
    c_get.add_argument("key", choices=known_keys(), metavar="KEY")
    c_get.set_defaults(func=cmd_config_get)

    c_set = csub.add_parser("set", help="Set a configuration value and save.")
    c_set.add_argument("key", choices=known_keys(), metavar="KEY")
    c_set.add_argument("value", help="New value (use 'none' to clear optional keys).")
    c_set.set_defaults(func=cmd_config_set)

    c_unset = csub.add_parser("unset", help="Reset a key to its built-in default.")
    c_unset.add_argument("key", choices=known_keys(), metavar="KEY")
    c_unset.set_defaults(func=cmd_config_unset)

    p_doctor = sub.add_parser(
        "doctor", help="Check whether this machine is ready to run LLMs."
    )
    p_doctor.add_argument("--json", action="store_true", help="Emit JSON.")
    p_doctor.set_defaults(func=cmd_doctor)

    p_download = sub.add_parser(
        "download", help="Pre-download a model from Hugging Face into the local cache."
    )
    p_download.add_argument(
        "model", nargs="?", default=config.model,
        help="HF repo id." + _model_default_note(config),
    )
    p_download.set_defaults(func=cmd_download)

    p_setup = sub.add_parser(
        "setup",
        help="Onboard this Mac: check readiness, pick + download a model, set it as default.",
    )
    p_setup.add_argument(
        "--model", "-m", default=None,
        help="Skip the recommendation and use this repo id.",
    )
    p_setup.add_argument(
        "--safety", type=float, default=config.safety_fraction,
        help="Fraction of memory usable when recommending a model (0-1].",
    )
    p_setup.add_argument(
        "--no-download", action="store_true", help="Set the default but don't fetch weights."
    )
    p_setup.add_argument(
        "--no-smoke-test", action="store_true", help="Skip the test generation."
    )
    p_setup.add_argument(
        "--force", action="store_true",
        help="Continue past failed readiness checks (e.g. to only set config).",
    )
    p_setup.add_argument("--trust-remote-code", action="store_true")
    p_setup.set_defaults(func=cmd_setup)

    return parser


def _model_default_note(config: UserConfig) -> str:
    return f" (default from config: {config.model})" if config.model else ""


def cmd_info(args) -> int:
    hw = detect_hardware()
    if args.json:
        print(json.dumps(hw.to_dict(), indent=2))
        return 0
    print(f"System:            {hw.system} ({hw.machine})")
    print(f"Chip:              {hw.chip or 'unknown'}")
    print(f"OS:                {hw.os_version or 'unknown'}")
    cores = hw.cpu_cores or "?"
    pe = ""
    if hw.performance_cores or hw.efficiency_cores:
        pe = f" ({hw.performance_cores or '?'}P + {hw.efficiency_cores or '?'}E)"
    print(f"CPU cores:         {cores}{pe}")
    print(f"GPU cores:         {hw.gpu_cores if hw.gpu_cores is not None else 'unknown'}")
    print(f"Total RAM:         {format_bytes(hw.total_ram_bytes)}")
    print(f"Recommended budget:{format_bytes(hw.recommended_working_set_bytes)}")
    print(f"MLX importable:    {hw.mlx_available}")
    print(f"Can run mlx-lm:    {hw.can_run_mlx}")
    if not hw.can_run_mlx:
        print(
            "\n⚠  This is not an Apple-silicon Mac; mlx-lm cannot execute here.",
            file=sys.stderr,
        )
    return 0


def cmd_fit(args) -> int:
    try:
        num_params = parse_param_count(args.params)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    hw = detect_hardware()
    available = hw.recommended_working_set_bytes or hw.total_ram_bytes
    if not available:
        print("error: could not determine available memory.", file=sys.stderr)
        return 2

    est = estimate_model_memory(
        num_params,
        quant_bits=args.bits,
        dtype=args.dtype,
        group_size=args.group_size,
        num_layers=args.layers,
        num_kv_heads=args.kv_heads,
        head_dim=args.head_dim,
        seq_len=args.seq_len,
    )
    fit = check_fit(est, available, safety_fraction=args.safety)
    rec = recommend_quantization(num_params, available, safety_fraction=args.safety)

    if args.json:
        print(
            json.dumps(
                {
                    "num_params": num_params,
                    "estimate": est.to_dict(),
                    "fit": fit.to_dict(),
                    "recommended": rec,
                    "available_bytes": available,
                },
                indent=2,
            )
        )
        return 0 if fit.fits else 1

    label = f"{num_params / 1e9:.2f}B params" if num_params >= 1e9 else f"{num_params:,} params"
    quant = f"{args.bits}-bit" if args.bits else f"{args.dtype}"
    print(f"Model:     {label} @ {quant}")
    print(f"Estimate:  {est.human()}")
    print(f"Memory:    {fit.human()}")
    rec_label = f"{rec['quant_bits']}-bit" if rec["quant_bits"] else "full precision (fp16)"
    if rec["fits"]:
        print(f"Best fit:  {rec_label} ({format_bytes(rec['weights_bytes'])} weights)")
    else:
        print("Best fit:  even 2-bit weights exceed the budget on this machine.")
    return 0 if fit.fits else 1


def _load_runner_or_exit(args):
    from .runner import MLXNotAvailableError, ModelRunner

    try:
        return ModelRunner.load(
            args.model,
            adapter_path=args.adapter_path,
            trust_remote_code=args.trust_remote_code,
        )
    except MLXNotAvailableError as e:
        print(f"error: {e}", file=sys.stderr)
        raise SystemExit(3)


def cmd_generate(args) -> int:
    prompt = args.prompt
    if prompt is None:
        prompt = sys.stdin.read().strip()
    if not prompt:
        print("error: no prompt provided (pass --prompt or pipe stdin).", file=sys.stderr)
        return 2

    config = _config_from_args(args)
    runner = _load_runner_or_exit(args)

    prompt_cache = None
    if args.prompt_cache_file:
        prompt_cache = runner.load_prompt_cache(args.prompt_cache_file)

    if args.no_stream:
        print(
            runner.generate(
                prompt=prompt, system=args.system, config=config,
                prompt_cache=prompt_cache,
            )
        )
    else:
        for delta in runner.stream(
            prompt=prompt, system=args.system, config=config,
            prompt_cache=prompt_cache,
        ):
            sys.stdout.write(delta)
            sys.stdout.flush()
        sys.stdout.write("\n")

    if args.stats and runner.last_stats:
        print(f"[{runner.last_stats.human()}]", file=sys.stderr)
    return 0


def cmd_cache(args) -> int:
    context = args.context
    if context is None:
        context = sys.stdin.read()
    if not context.strip():
        print("error: no context provided (pass --context or pipe stdin).", file=sys.stderr)
        return 2

    runner = _load_runner_or_exit(args)
    runner.build_and_save_prompt_cache(context, args.out, max_kv_size=args.max_kv_size)
    print(f"Saved prompt cache to {args.out}", file=sys.stderr)
    return 0


def _launch_tunnel(port: int):
    """Start a Cloudflare quick tunnel to localhost:port; print the public URL.

    Returns the cloudflared Popen (or None if cloudflared isn't installed). A
    background thread scans its output for the trycloudflare.com URL and prints it.
    """
    import shutil

    if not shutil.which("cloudflared"):
        print(
            "warning: --tunnel requested but `cloudflared` is not installed; serving "
            "locally only. Install it with `brew install cloudflared`.",
            file=sys.stderr,
        )
        return None

    proc = subprocess.Popen(
        ["cloudflared", "tunnel", "--url", f"http://127.0.0.1:{port}"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
    )

    def _watch():
        import re

        pat = re.compile(r"https://[-\w.]+\.trycloudflare\.com")
        for line in proc.stdout:  # type: ignore[union-attr]
            m = pat.search(line)
            if m:
                print(f"\n  Public URL: {m.group(0)}/v1/messages\n", file=sys.stderr)
                break

    threading.Thread(target=_watch, daemon=True).start()
    return proc


def cmd_serve(args) -> int:
    hw = detect_hardware()
    if not hw.can_run_mlx:
        print(
            "warning: this is not an Apple-silicon Mac; the model may not run.",
            file=sys.stderr,
        )

    tunnel = _launch_tunnel(args.port) if args.tunnel else None
    try:
        if args.api == "anthropic":
            return _serve_anthropic(args)
        return _serve_openai(args)
    finally:
        if tunnel is not None:
            tunnel.terminate()


def _serve_anthropic(args) -> int:
    if not args.model:
        print(
            "error: serve --api anthropic needs a model (pass --model or set a default "
            "with `mlx-runner config set model ...`).",
            file=sys.stderr,
        )
        return 2
    from .anthropic_server import serve as serve_anthropic

    print(
        f"Serving Anthropic Messages API on http://{args.host}:{args.port}/v1/messages  "
        "(Ctrl-C to stop)",
        file=sys.stderr,
    )
    try:
        return serve_anthropic(
            args.model, args.host, args.port,
            adapter_path=args.adapter_path,
            trust_remote_code=args.trust_remote_code,
            api_key=args.api_key,
        )
    except KeyboardInterrupt:
        return 0


def _serve_openai(args) -> int:
    cmd = [sys.executable, "-m", "mlx_lm.server", "--host", args.host, "--port", str(args.port)]
    if args.model:
        cmd += ["--model", args.model]
    if args.adapter_path:
        cmd += ["--adapter-path", args.adapter_path]
    if args.trust_remote_code:
        cmd += ["--trust-remote-code"]
    # argparse.REMAINDER keeps a leading "--"; drop it before forwarding.
    extra = [a for a in (args.server_args or []) if a != "--"]
    cmd += extra

    print(f"Serving OpenAI-compatible API on http://{args.host}:{args.port}/v1  (Ctrl-C to stop)", file=sys.stderr)
    try:
        return subprocess.call(cmd)
    except FileNotFoundError:
        print(
            "error: could not launch mlx_lm.server. Install it with `pip install mlx-lm`.",
            file=sys.stderr,
        )
        return 3
    except KeyboardInterrupt:
        return 0


def cmd_chat(args) -> int:
    config = _config_from_args(args)
    runner = _load_runner_or_exit(args)

    messages: List[dict] = []
    if args.system:
        messages.append({"role": "system", "content": args.system})

    print("Chat ready. Type 'exit'/'quit' or press Ctrl-D to leave.")
    while True:
        try:
            user = input("\nyou> ").strip()
        except EOFError:
            print()
            break
        if user.lower() in {"exit", "quit"}:
            break
        if not user:
            continue
        messages.append({"role": "user", "content": user})
        sys.stdout.write("bot> ")
        parts: List[str] = []
        for delta in runner.stream(messages=messages, config=config):
            parts.append(delta)
            sys.stdout.write(delta)
            sys.stdout.flush()
        sys.stdout.write("\n")
        messages.append({"role": "assistant", "content": "".join(parts)})
    return 0


def cmd_config_show(args) -> int:
    config = load_config()
    if args.json:
        print(json.dumps(config.to_dict(), indent=2))
        return 0
    print(f"# {default_config_path()}")
    for key in known_keys():
        value = getattr(config, key)
        print(f"{key} = {value if value is not None else '(default)'}")
    return 0


def cmd_config_path(args) -> int:
    print(default_config_path())
    return 0


def cmd_config_get(args) -> int:
    value = getattr(load_config(), args.key)
    print("" if value is None else value)
    return 0


def cmd_config_set(args) -> int:
    try:
        value = coerce_value(args.key, args.value)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    config = load_config()
    setattr(config, args.key, value)
    path = save_config(config)
    print(f"{args.key} = {value if value is not None else '(default)'}  -> {path}")
    return 0


def cmd_config_unset(args) -> int:
    config = load_config()
    setattr(config, args.key, getattr(UserConfig(), args.key))
    path = save_config(config)
    print(f"{args.key} reset to default  -> {path}")
    return 0


_STATUS_MARK = {OK: "✓", WARN: "!", FAIL: "✗"}


def _print_checks(checks: List[Check]) -> None:
    for c in checks:
        mark = _STATUS_MARK.get(c.status, "?")
        print(f"  {mark} {c.name}: {c.detail}")
        if c.status != OK and c.remediation:
            print(f"      → {c.remediation}")


def cmd_doctor(args) -> int:
    checks = run_checks()
    if args.json:
        print(json.dumps([c.to_dict() for c in checks], indent=2))
    else:
        print("Readiness checks:")
        _print_checks(checks)
        ready = is_ready(checks)
        print("\n" + ("Ready to run LLMs." if ready else "Not ready — resolve the ✗ items above."))
    return 0 if is_ready(checks) else 1


def _download_model(repo_id: str) -> str:
    """Fetch a model snapshot into the local HF cache; return its path.

    ``huggingface_hub`` ships with mlx-lm; import it lazily so the rest of the
    CLI works without it.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise RuntimeError(
            "huggingface_hub is not installed. Install it with `pip install mlx-lm` "
            "(it is pulled in as a dependency)."
        ) from e
    return snapshot_download(repo_id)


def cmd_download(args) -> int:
    repo_id = args.model
    if not repo_id:
        print(
            "error: no model given and no default configured "
            "(pass a repo id or run `mlx-runner config set model ...`).",
            file=sys.stderr,
        )
        return 2
    print(f"Downloading {repo_id} ...", file=sys.stderr)
    try:
        path = _download_model(repo_id)
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 3
    except Exception as e:  # network / repo errors from hub
        print(f"error: download failed: {e}", file=sys.stderr)
        return 1
    print(path)
    return 0


def cmd_setup(args) -> int:
    hw = detect_hardware()

    # 1. Readiness.
    print("Checking readiness ...")
    checks = run_checks(hw)
    _print_checks(checks)
    if not is_ready(checks) and not args.force:
        print(
            "\nNot ready — resolve the ✗ items above, or re-run with --force to "
            "configure anyway (download/smoke-test will likely fail).",
            file=sys.stderr,
        )
        return 1

    # 2. Choose a model.
    model = args.model
    if model:
        print(f"\nUsing requested model: {model}")
    else:
        available = hw.recommended_working_set_bytes or hw.total_ram_bytes
        if not available:
            print("error: could not determine available memory to recommend a model.", file=sys.stderr)
            return 2
        rec = recommend_model(available, safety_fraction=args.safety)
        if rec is None:
            print(
                "error: no catalog model fits this machine's memory budget. "
                "Specify a tiny model explicitly with --model.",
                file=sys.stderr,
            )
            return 1
        model = rec.repo_id
        fit = rec.estimate()
        print(f"\nRecommended: {rec.name}")
        print(f"  {model}")
        print(f"  ~{rec.params / 1e9:.1f}B params @ {rec.quant_bits}-bit, weights {fit.human()}")

    # 3. Download (unless skipped).
    if args.no_download:
        print("\nSkipping download (--no-download).")
    else:
        print(f"\nDownloading {model} (this can take a while the first time) ...")
        try:
            path = _download_model(model)
            print(f"  cached at {path}")
        except RuntimeError as e:
            print(f"error: {e}", file=sys.stderr)
            return 3
        except Exception as e:
            print(f"error: download failed: {e}", file=sys.stderr)
            return 1

    # 4. Persist as the default model.
    cfg = load_config()
    cfg.model = model
    cfg_path = save_config(cfg)
    print(f"\nSet default model -> {model}\n  ({cfg_path})")

    # 5. Smoke test.
    if args.no_smoke_test:
        print("\nSkipping smoke test (--no-smoke-test).")
    elif args.no_download:
        print("\nSkipping smoke test (no weights downloaded).")
    else:
        print("\nRunning a smoke-test generation ...")
        try:
            from .runner import GenerationConfig, MLXNotAvailableError, ModelRunner

            runner = ModelRunner.load(model, trust_remote_code=args.trust_remote_code)
            out = runner.generate(
                prompt="Reply with a single short sentence to confirm you are working.",
                config=GenerationConfig(max_tokens=32, temperature=0.0),
            )
            print(f"  model said: {out.strip()[:200]}")
        except MLXNotAvailableError as e:
            print(f"  smoke test skipped: {e}", file=sys.stderr)
            return 3
        except Exception as e:
            print(f"  smoke test failed: {e}", file=sys.stderr)
            return 1

    print("\n✓ Setup complete. Try:  mlx-runner chat")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args) or 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
