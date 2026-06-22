from __future__ import annotations

import sys
from typing import List

from ._config_from_args import _config_from_args
from ._load_runner_or_exit import _load_runner_or_exit


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
