from __future__ import annotations


class _AnthropicErrorBodyMixin:
    def body(self) -> dict:
        return {"type": "error", "error": {"type": self.error_type, "message": self.message}}
