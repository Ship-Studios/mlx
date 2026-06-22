from __future__ import annotations

import sys

from ._load_runner_or_exit import _load_runner_or_exit


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
