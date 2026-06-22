"""An Anthropic Messages API-compatible HTTP server backed by a local MLX model.

Emulates ``POST /v1/messages`` (non-streaming and SSE streaming) so that Anthropic
SDK clients can talk to a model served by :class:`mlx_runner.runner.ModelRunner`.
The exact wire format is pinned by the vendored type stubs and ``WIRE_FORMAT.md``
under ``reference/anthropic-api/``.

The request parsing, response building, SSE event formatting, and stop-sequence
handling are pure-Python and dependency-free (unit-tested without mlx). Only
``serve()`` and the generation glue touch the Apple-silicon-only model, and those
import :mod:`mlx_runner.runner` lazily.
"""
from __future__ import annotations

from ._constants import ANTHROPIC_VERSION, MAX_REQUEST_BYTES
from .anthropic_error import AnthropicError
from .parsed_request import ParsedRequest
from ._content_to_text import _content_to_text
from .parse_request import parse_request
from .stop_sequence_filter import StopSequenceFilter
from .new_message_id import new_message_id
from .build_message import build_message
from .sse_event import sse_event
from .message_start_event import message_start_event
from .content_block_start_event import content_block_start_event
from .ping_event import ping_event
from .content_block_delta_event import content_block_delta_event
from .content_block_stop_event import content_block_stop_event
from .message_delta_event import message_delta_event
from .message_stop_event import message_stop_event
from ._generation_config import _generation_config
from ._count_tokens import _count_tokens
from ._output_tokens import _output_tokens
from ._input_tokens import _input_tokens
from .generate_full import generate_full
from .generate_stream import generate_stream
from .make_handler import make_handler
from ._constant_time_eq import _constant_time_eq
from .serve import serve

__all__ = [
    "ANTHROPIC_VERSION",
    "MAX_REQUEST_BYTES",
    "AnthropicError",
    "ParsedRequest",
    "_content_to_text",
    "parse_request",
    "StopSequenceFilter",
    "new_message_id",
    "build_message",
    "sse_event",
    "message_start_event",
    "content_block_start_event",
    "ping_event",
    "content_block_delta_event",
    "content_block_stop_event",
    "message_delta_event",
    "message_stop_event",
    "_generation_config",
    "_count_tokens",
    "_output_tokens",
    "_input_tokens",
    "generate_full",
    "generate_stream",
    "make_handler",
    "_constant_time_eq",
    "serve",
]
