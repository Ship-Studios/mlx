from __future__ import annotations

from typing import List, Optional

from .build_parser import build_parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args) or 0
