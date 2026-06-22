from __future__ import annotations

from typing import Optional

from ._import_cache_module import _import_cache_module


class _ModelRunnerMakePromptCacheMixin:
    def make_prompt_cache(self, max_kv_size: Optional[int] = None):
        """Create a fresh, empty KV prompt cache bound to this model."""
        cache_mod = _import_cache_module()
        return cache_mod.make_prompt_cache(self.model, max_kv_size=max_kv_size)
