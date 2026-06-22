# Anthropic Messages API — wire format (for the `serve --api anthropic` emulator)

Distilled from the Stainless-generated type stubs vendored in this directory
(`anthropic-sdk-python`, `src/anthropic/types/`). These are the exact shapes our
local server must **accept** and **emit** so Anthropic SDK clients work unchanged
against an MLX model. Source of truth is the `.py` files here; this is the summary.

## Endpoint

`POST /v1/messages` — body is JSON. Auth header is `x-api-key` (Anthropic) — our
server accepts but does not require it (configurable). Also send
`anthropic-version: 2023-06-01` (clients set it; we ignore the value).

## Request body (`MessageCreateParams`)

Required: `max_tokens` (int), `messages` (array), `model` (string).

```jsonc
{
  "model": "string",                  // required (we map/ignore; serves the loaded model)
  "max_tokens": 1024,                 // required
  "messages": [                       // required; roles alternate user/assistant
    {"role": "user", "content": "hi"},                         // content: string, OR
    {"role": "user", "content": [{"type": "text", "text": "hi"}]}  // array of blocks
  ],
  "system": "string OR [{type:text,text}]",  // optional
  "stop_sequences": ["</done>"],      // optional
  "temperature": 1.0,                  // optional (0..1)
  "top_p": 0.0,                        // optional
  "top_k": 0,                          // optional
  "stream": false,                     // optional; true → SSE (below)
  "metadata": {"user_id": "..."}      // optional; ignored
}
```

We support text content only. Tool/think/image blocks may be present in the type
system but a local text model does not handle them — reject with 400
`invalid_request_error` rather than silently dropping (matches Anthropic's behavior
for unsupported params).

## Non-streaming response (`Message`)

```jsonc
{
  "id": "msg_<random>",
  "type": "message",                  // literal
  "role": "assistant",                // literal
  "model": "<model id>",
  "content": [{"type": "text", "text": "..."}],
  "stop_reason": "end_turn",          // end_turn | max_tokens | stop_sequence (also tool_use|pause_turn|refusal upstream)
  "stop_sequence": null,              // the matched custom stop sequence, else null
  "usage": {"input_tokens": 12, "output_tokens": 34}
}
```

`usage` may also carry `cache_creation_input_tokens` / `cache_read_input_tokens`
(nullable) and `service_tier`; for a local model emit just `input_tokens` and
`output_tokens`.

## Streaming (`stream: true`) — Server-Sent Events

`Content-Type: text/event-stream`. Each event is:

```
event: <type>\n
data: <one-line JSON>\n
\n
```

Emit this exact sequence (text-only case):

1. **`message_start`** — full `Message` with `content: []`, `stop_reason: null`,
   `usage.input_tokens` set and `output_tokens: 0`.
   ```json
   {"type":"message_start","message":{"id":"msg_...","type":"message","role":"assistant","model":"...","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":12,"output_tokens":0}}}
   ```
2. **`content_block_start`** (index 0, empty text block)
   ```json
   {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}
   ```
3. **`ping`** — optional keep-alive; Anthropic sends one or more. `{"type":"ping"}`
4. **`content_block_delta`** — one per generated chunk, `delta.type` = `text_delta`
   ```json
   {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hel"}}
   ```
5. **`content_block_stop`** (index 0) — `{"type":"content_block_stop","index":0}`
6. **`message_delta`** — final stop reason + cumulative output token count
   ```json
   {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":34}}
   ```
7. **`message_stop`** — `{"type":"message_stop"}`

The SDK accumulates these into the same `Message` shown above, so the field names
and the event order must match exactly.

## stop_reason mapping (local generation → API)

| Local outcome | `stop_reason` | `stop_sequence` |
| --- | --- | --- |
| Model emitted its natural end / EOS | `end_turn` | `null` |
| Hit `max_tokens` | `max_tokens` | `null` |
| Output contained a `stop_sequences` entry | `stop_sequence` | the matched string |

## Errors (Anthropic envelope)

```json
{"type":"error","error":{"type":"invalid_request_error","message":"..."}}
```

`error.type` ∈ `invalid_request_error` (400), `authentication_error` (401),
`not_found_error` (404), `rate_limit_error` (429), `api_error` (500),
`overloaded_error` (529).
