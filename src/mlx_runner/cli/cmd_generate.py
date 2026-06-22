from __future__ import annotations

import sys

from ._config_from_args import _config_from_args
from ._load_runner_or_exit import _load_runner_or_exit


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
