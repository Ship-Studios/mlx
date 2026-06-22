from __future__ import annotations

import secrets


def new_message_id() -> str:
    return "msg_" + secrets.token_hex(12)
