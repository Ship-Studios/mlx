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

import subprocess

from ._add_generation_args import _add_generation_args
from ._config_from_args import _config_from_args
from .build_parser import build_parser
from ._model_default_note import _model_default_note
from .cmd_info import cmd_info
from .cmd_fit import cmd_fit
from ._load_runner_or_exit import _load_runner_or_exit
from .cmd_generate import cmd_generate
from .cmd_cache import cmd_cache
from ._launch_tunnel import _launch_tunnel
from .cmd_serve import cmd_serve
from ._serve_anthropic import _serve_anthropic
from ._serve_openai import _serve_openai
from .cmd_chat import cmd_chat
from .cmd_config_show import cmd_config_show
from .cmd_config_path import cmd_config_path
from .cmd_config_get import cmd_config_get
from .cmd_config_set import cmd_config_set
from .cmd_config_unset import cmd_config_unset
from ._constants import _STATUS_MARK
from ._print_checks import _print_checks
from .cmd_doctor import cmd_doctor
from ._download_model import _download_model
from .cmd_download import cmd_download
from .cmd_setup import cmd_setup
from .main import main

__all__ = [
    "_add_generation_args",
    "_config_from_args",
    "build_parser",
    "_model_default_note",
    "cmd_info",
    "cmd_fit",
    "_load_runner_or_exit",
    "cmd_generate",
    "cmd_cache",
    "_launch_tunnel",
    "cmd_serve",
    "_serve_anthropic",
    "_serve_openai",
    "cmd_chat",
    "cmd_config_show",
    "cmd_config_path",
    "cmd_config_get",
    "cmd_config_set",
    "cmd_config_unset",
    "_STATUS_MARK",
    "_print_checks",
    "cmd_doctor",
    "_download_model",
    "cmd_download",
    "cmd_setup",
    "main",
]
