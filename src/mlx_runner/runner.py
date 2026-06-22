"""A thin, hardware-aware wrapper around mlx-lm for loading models and generating text.

``mlx_lm`` (and ``mlx``) are imported lazily, inside the methods that need them,
so this module — and the whole ``mlx_runner`` package — imports cleanly on any
platform. The heavy, Apple-silicon-only dependency is only required when you
actually load a model or generate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional


class MLXNotAvailableError(RuntimeError):
    """Raised when mlx-lm is needed but cannot be imported (e.g. non Apple silicon)."""


def _import_mlx_lm():
    try:
        import mlx_lm  # noqa: F401

        return mlx_lm
    except ImportError as e:  # pragma: no cover - exercised via monkeypatch in tests
        raise MLXNotAvailableError(
            "mlx-lm is not installed or not importable. Install it with "
            "`pip install mlx-lm` on an Apple-silicon Mac."
        ) from e


def _import_cache_module():
    """Lazily import mlx-lm's prompt-cache helpers."""
    _import_mlx_lm()  # surface a friendly error if mlx-lm is missing
    from mlx_lm.models import cache as cache_mod

    return cache_mod


@dataclass
class GenerationConfig:
    """Sampling and decoding parameters for a generation request."""

    max_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 0.0
    top_k: int = 0
    min_p: float = 0.0
    min_tokens_to_keep: int = 1
    repetition_penalty: Optional[float] = None
    repetition_context_size: int = 20
    logit_bias: Optional[Dict[int, float]] = None
    seed: Optional[int] = None
    max_kv_size: Optional[int] = None
    kv_bits: Optional[int] = None


@dataclass
class GenerationStats:
    """Timing and memory statistics reported by mlx-lm after a generation."""

    prompt_tokens: int = 0
    generation_tokens: int = 0
    prompt_tps: float = 0.0
    generation_tps: float = 0.0
    peak_memory_gb: float = 0.0
    finish_reason: Optional[str] = None

    def human(self) -> str:
        return (
            f"prompt {self.prompt_tokens} tok @ {self.prompt_tps:.1f} tok/s | "
            f"gen {self.generation_tokens} tok @ {self.generation_tps:.1f} tok/s | "
            f"peak {self.peak_memory_gb:.2f} GB"
            + (f" | {self.finish_reason}" if self.finish_reason else "")
        )


class ModelRunner:
    """Loads an mlx-lm model once and serves repeated generations from it."""

    def __init__(self, model, tokenizer, model_path: Optional[str] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.model_path = model_path
        self.last_stats: Optional[GenerationStats] = None

    @classmethod
    def load(
        cls,
        model_path: str,
        adapter_path: Optional[str] = None,
        tokenizer_config: Optional[dict] = None,
        trust_remote_code: bool = False,
        lazy: bool = False,
    ) -> "ModelRunner":
        """Load a model + tokenizer from an HF repo id or local path."""
        mlx_lm = _import_mlx_lm()
        model, tokenizer = mlx_lm.load(
            model_path,
            adapter_path=adapter_path,
            tokenizer_config=tokenizer_config or {},
            trust_remote_code=trust_remote_code,
            lazy=lazy,
        )
        return cls(model, tokenizer, model_path=model_path)

    # -- prompt caching ------------------------------------------------------

    def make_prompt_cache(self, max_kv_size: Optional[int] = None):
        """Create a fresh, empty KV prompt cache bound to this model."""
        cache_mod = _import_cache_module()
        return cache_mod.make_prompt_cache(self.model, max_kv_size=max_kv_size)

    def load_prompt_cache(self, path: str):
        """Load a previously saved prompt cache (``.safetensors``) from disk."""
        cache_mod = _import_cache_module()
        return cache_mod.load_prompt_cache(path)

    def save_prompt_cache(self, path: str, prompt_cache) -> None:
        """Persist a prompt cache to disk for later reuse."""
        cache_mod = _import_cache_module()
        cache_mod.save_prompt_cache(path, prompt_cache)

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

    # -- prompt formatting ---------------------------------------------------

    def format_prompt(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[dict]] = None,
        system: Optional[str] = None,
        add_generation_prompt: bool = True,
    ):
        """Build a model-ready prompt, applying the tokenizer's chat template.

        Returns token ids (the tokenizer's chat template tokenizes by default);
        mlx-lm's generate accepts either token ids or a raw string. Falls back to
        plain text when the tokenizer has no chat template.
        """
        if messages is None:
            if prompt is None:
                raise ValueError("provide either `prompt` or `messages`")
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})

        chat_template = getattr(self.tokenizer, "chat_template", None)
        if chat_template:
            return self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=add_generation_prompt
            )
        # No chat template: concatenate message contents as a best-effort prompt.
        return "\n".join(str(m.get("content", "")) for m in messages)

    # -- generation ----------------------------------------------------------

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

    @staticmethod
    def _apply_seed(config: GenerationConfig) -> None:
        if config.seed is None:
            return
        try:
            import mlx.core as mx

            mx.random.seed(config.seed)
        except ImportError:  # pragma: no cover - mlx absent
            pass

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
