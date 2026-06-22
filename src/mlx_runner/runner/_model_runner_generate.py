from __future__ import annotations

from typing import List, Optional

from .generation_config import GenerationConfig


class _ModelRunnerGenerateMixin:
    def generate(
        self,
        prompt: Optional[str] = None,
        *,
        messages: Optional[List[dict]] = None,
        system: Optional[str] = None,
        config: Optional[GenerationConfig] = None,
        prompt_cache=None,
    ) -> str:
        """Generate and return the full completion text (non-streaming)."""
        return "".join(
            self.stream(
                prompt=prompt,
                messages=messages,
                system=system,
                config=config,
                prompt_cache=prompt_cache,
            )
        )
