from __future__ import annotations

from ._import_cache_module import _import_cache_module


class _ModelRunnerSavePromptCacheMixin:
    def save_prompt_cache(self, path: str, prompt_cache) -> None:
        """Persist a prompt cache to disk for later reuse."""
        cache_mod = _import_cache_module()
        cache_mod.save_prompt_cache(path, prompt_cache)
