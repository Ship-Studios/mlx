"""Command-line interface for mlx_runner.

Subcommands:
  info      Show detected hardware and MLX availability.
  fit       Estimate whether a model of N parameters fits in memory.
  generate  Generate text from a prompt (alias: run).
  chat      Interactive chat REPL.
  cache     Pre-compute and save a reusable prompt (KV) cache for a long context.
  serve     Launch an OpenAI-compatible HTTP server (mlx_lm.server).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from typing import List, Optional

from . import __version__
from .hardware import detect_hardware
from .memory import (
    check_fit,
    estimate_model_memory,
    format_bytes,
    parse_param_count,
    recommend_quantization,
)


def _add_generation_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("generation")
    g.add_argument("--max-tokens", type=int, default=512)
    g.add_argument(
        "--temp", "--temperature", dest="temperature", type=float, default=0.0
    )
    g.add_argument("--top-p", type=float, default=0.0)
    g.add_argument("--top-k", type=int, default=0)
    g.add_argument("--min-p", type=float, default=0.0)
    g.add_argument("--repetition-penalty", type=float, default=None)
    g.add_argument("--seed", type=int, default=None)
    g.add_argument("--max-kv-size", type=int, default=None)
    g.add_argument("--kv-bits", type=int, default=None)


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


def build_parser() -> argparse.ArgumentParser:
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
        "--safety", type=float, default=0.9, help="Fraction of memory usable (0-1]."
    )
    p_fit.add_argument("--json", action="store_true")
    p_fit.set_defaults(func=cmd_fit)

    p_gen = sub.add_parser(
        "generate", aliases=["run"], help="Generate text from a prompt."
    )
    p_gen.add_argument("--model", "-m", required=True, help="HF repo id or local path.")
    p_gen.add_argument("--prompt", "-p", help="User prompt; reads stdin if omitted.")
    p_gen.add_argument("--system", default=None, help="Optional system prompt.")
    p_gen.add_argument("--adapter-path", default=None, help="Optional LoRA adapter path.")
    p_gen.add_argument("--no-stream", action="store_true", help="Disable token streaming.")
    p_gen.add_argument("--stats", action="store_true", help="Print stats to stderr.")
    p_gen.add_argument("--trust-remote-code", action="store_true")
    p_gen.add_argument(
        "--prompt-cache-file",
        default=None,
        help="Reuse a prompt cache saved by `cache` so the context isn't re-encoded.",
    )
    _add_generation_args(p_gen)
    p_gen.set_defaults(func=cmd_generate)

    p_chat = sub.add_parser("chat", help="Interactive chat REPL.")
    p_chat.add_argument("--model", "-m", required=True, help="HF repo id or local path.")
    p_chat.add_argument("--system", default=None, help="Optional system prompt.")
    p_chat.add_argument("--adapter-path", default=None)
    p_chat.add_argument("--trust-remote-code", action="store_true")
    _add_generation_args(p_chat)
    p_chat.set_defaults(func=cmd_chat)

    p_cache = sub.add_parser(
        "cache", help="Pre-compute a reusable prompt (KV) cache for a long context."
    )
    p_cache.add_argument("--model", "-m", required=True, help="HF repo id or local path.")
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
        "serve", help="Launch an OpenAI-compatible HTTP server (mlx_lm.server)."
    )
    p_serve.add_argument("--model", "-m", default=None, help="Default model to serve.")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8080)
    p_serve.add_argument("--adapter-path", default=None)
    p_serve.add_argument("--trust-remote-code", action="store_true")
    p_serve.add_argument(
        "server_args",
        nargs=argparse.REMAINDER,
        help="Extra args forwarded verbatim to mlx_lm.server (after --).",
    )
    p_serve.set_defaults(func=cmd_serve)

    return parser


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


def cmd_serve(args) -> int:
    hw = detect_hardware()
    if not hw.can_run_mlx:
        print(
            "warning: this is not an Apple-silicon Mac; mlx_lm.server may not run.",
            file=sys.stderr,
        )

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

    print(f"Serving on http://{args.host}:{args.port}/v1  (Ctrl-C to stop)", file=sys.stderr)
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


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args) or 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
