# Vendored reference: Anthropic Messages API type stubs

The `*.py` files in this directory are copied verbatim from the
[anthropic-sdk-python](https://github.com/anthropics/anthropic-sdk-python) project
(`src/anthropic/types/`, `main` branch), which is MIT-licensed (© Anthropic, PBC).

They are vendored **for reference only** — to pin the exact request/response and
streaming-event wire format that `mlx-runner serve --api anthropic` emulates. They
are not imported or executed by `mlx_runner`; the package has no runtime dependency
on the `anthropic` SDK. `WIRE_FORMAT.md` is our distilled summary of these files.

To refresh: re-download the same filenames from
`https://raw.githubusercontent.com/anthropics/anthropic-sdk-python/main/src/anthropic/types/`.
