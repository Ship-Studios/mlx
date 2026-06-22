from __future__ import annotations

from .generation_config import GenerationConfig


class _ModelRunnerBuildKwargsMixin:
    def _build_kwargs(self, config: GenerationConfig) -> dict:
        from mlx_lm.sample_utils import make_logits_processors, make_sampler

        sampler = make_sampler(
            temp=config.temperature,
            top_p=config.top_p,
            min_p=config.min_p,
            min_tokens_to_keep=config.min_tokens_to_keep,
            top_k=config.top_k,
        )
        logits_processors = make_logits_processors(
            logit_bias=config.logit_bias,
            repetition_penalty=config.repetition_penalty,
            repetition_context_size=config.repetition_context_size,
        )
        kwargs = {
            "max_tokens": config.max_tokens,
            "sampler": sampler,
            "logits_processors": logits_processors,
        }
        if config.max_kv_size is not None:
            kwargs["max_kv_size"] = config.max_kv_size
        if config.kv_bits is not None:
            kwargs["kv_bits"] = config.kv_bits
        return kwargs
