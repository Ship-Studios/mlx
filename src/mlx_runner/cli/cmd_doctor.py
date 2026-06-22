from __future__ import annotations

import json

from ..doctor import is_ready, run_checks
from ._print_checks import _print_checks


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
