from __future__ import annotations

from ._anthropic_error_init import _AnthropicErrorInitMixin
from ._anthropic_error_body import _AnthropicErrorBodyMixin


class AnthropicError(_AnthropicErrorInitMixin, _AnthropicErrorBodyMixin, Exception):
    """An error renderable as the Anthropic error envelope with an HTTP status."""
