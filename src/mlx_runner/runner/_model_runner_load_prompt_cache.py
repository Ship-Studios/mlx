from __future__ import annotations

from ._import_cache_module import _import_cache_module


class _ModelRunnerLoadPromptCacheMixin:
    def load_prompt_cache(self, path: str):
        """Load a previously saved prompt cache (``.safetensors``) from disk."""
        cache_mod = _import_cache_module()
        return cache_mod.load_prompt_cache(path)
