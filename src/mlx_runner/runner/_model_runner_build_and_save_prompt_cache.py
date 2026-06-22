from __future__ import annotations

from typing import Optional


class _ModelRunnerBuildAndSavePromptCacheMixin:
    def build_and_save_prompt_cache(
        self,
        context: str,
        path: str,
        *,
        max_kv_size: Optional[int] = None,
    ) -> None:
        """Pre-compute the KV activations for ``context`` and write them to ``path``.

        The resulting file can be passed to :meth:`stream`/:meth:`generate` (or
        the CLI's ``--prompt-cache-file``) so the context is not re-encoded on
        every query — a large speedup for long, repeatedly-queried documents.
        """
        cache = self.make_prompt_cache(max_kv_size=max_kv_size)
        # Run the context through the model once to populate the cache; we
        # request a single token and discard it — only the cache state matters.
        for _ in self.stream(prompt=context, prompt_cache=cache):
            break
        self.save_prompt_cache(path, cache)
