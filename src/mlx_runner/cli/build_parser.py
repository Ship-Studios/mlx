from __future__ import annotations

import argparse
from typing import Optional

from .. import __version__
from ..config import UserConfig, known_keys, load_config
from ._add_generation_args import _add_generation_args
from ._model_default_note import _model_default_note
from .cmd_info import cmd_info
from .cmd_fit import cmd_fit
from .cmd_generate import cmd_generate
from .cmd_chat import cmd_chat
from .cmd_cache import cmd_cache
from .cmd_serve import cmd_serve
from .cmd_config_show import cmd_config_show
from .cmd_config_path import cmd_config_path
from .cmd_config_get import cmd_config_get
from .cmd_config_set import cmd_config_set
from .cmd_config_unset import cmd_config_unset
from .cmd_doctor import cmd_doctor
from .cmd_download import cmd_download
from .cmd_setup import cmd_setup


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
        "--api-key-file", default=None,
        help="anthropic: read the required x-api-key value from this file (e.g. a "
        "chmod-600 key file). Preferred for tunnels: unlike --api-key the secret never "
        "appears in argv (ps) or the process environment. --api-key takes precedence.",
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
