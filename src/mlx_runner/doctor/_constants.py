from __future__ import annotations

OK = "ok"
WARN = "warn"
FAIL = "fail"

# Minimum unified memory below which even the smallest 4-bit build is a squeeze.
_MIN_USABLE_RAM = 6 * 1024 ** 3
