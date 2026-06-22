from __future__ import annotations

from typing import Iterator, List, Optional

from ._import_mlx_lm import _import_mlx_lm
from .generation_config import GenerationConfig
from .generation_stats import GenerationStats


class _ModelRunnerStreamMixin:
    def stream(
        self,
        prompt: Optional[str] = None,
        *,
        messages: Optional[List[dict]] = None,
        system: Optional[str] = None,
        config: Optional[GenerationConfig] = None,
        prompt_cache=None,
    ) -> Iterator[str]:
        """Yield generated text deltas; updates ``self.last_stats`` when done.

        Pass ``prompt_cache`` (from :meth:`make_prompt_cache` or
        :meth:`load_prompt_cache`) to reuse pre-computed context activations.
        """
        mlx_lm = _import_mlx_lm()
        config = config or GenerationConfig()
        self._apply_seed(config)
        formatted = self.format_prompt(prompt=prompt, messages=messages, system=system)
        kwargs = self._build_kwargs(config)
        if prompt_cache is not None:
            kwargs["prompt_cache"] = prompt_cache

        last = None
        for response in mlx_lm.stream_generate(
            self.model, self.tokenizer, formatted, **kwargs
        ):
            last = response
            yield response.text

        if last is not None:
            self.last_stats = GenerationStats(
                prompt_tokens=int(getattr(last, "prompt_tokens", 0) or 0),
                generation_tokens=int(getattr(last, "generation_tokens", 0) or 0),
                prompt_tps=float(getattr(last, "prompt_tps", 0.0) or 0.0),
                generation_tps=float(getattr(last, "generation_tps", 0.0) or 0.0),
                peak_memory_gb=float(getattr(last, "peak_memory", 0.0) or 0.0),
                finish_reason=getattr(last, "finish_reason", None),
            )
