from __future__ import annotations

from typing import Optional

from ._import_mlx_lm import _import_mlx_lm


class _ModelRunnerLoadMixin:
    @classmethod
    def load(
        cls,
        model_path: str,
        adapter_path: Optional[str] = None,
        tokenizer_config: Optional[dict] = None,
        trust_remote_code: bool = False,
        lazy: bool = False,
    ) -> "ModelRunner":
        """Load a model + tokenizer from an HF repo id or local path.

        Args:
            model_path: Hugging Face repo id or local path to model weights.
            adapter_path: Optional LoRA adapter path (for fine-tuned models).
            tokenizer_config: Optional tokenizer configuration dict.
            trust_remote_code: Whether to trust the model's remote code.
            lazy: Whether to load lazily (deferred loading).

        Returns:
            A new ModelRunner instance with the loaded model and tokenizer.

        Raises:
            MLXNotAvailableError: If mlx-lm is not installed.
        """
        mlx_lm = _import_mlx_lm()
        # `trust_remote_code` only became a top-level `mlx_lm.load()` argument in
        # newer mlx-lm; older versions (at/above our >=0.21.0 floor) reject it with
        # TypeError. Try passing it; on the specific "unexpected keyword argument"
        # failure, fall back to the version-stable HF path — thread it through
        # tokenizer_config (omitting it entirely when False) — and retry. Signature
        # inspection alone isn't enough: a decorated/wrapped load can advertise
        # **kwargs yet still reject the arg, so we let the call itself be the oracle.
        tok_cfg = dict(tokenizer_config or {})
        load_kwargs = {"adapter_path": adapter_path, "tokenizer_config": tok_cfg, "lazy": lazy}
        try:
            model, tokenizer = mlx_lm.load(
                model_path, trust_remote_code=trust_remote_code, **load_kwargs
            )
        except TypeError as e:
            if "trust_remote_code" not in str(e):
                raise
            if trust_remote_code:
                tok_cfg["trust_remote_code"] = True  # same object as load_kwargs["tokenizer_config"]
            model, tokenizer = mlx_lm.load(model_path, **load_kwargs)
        return cls(model, tokenizer, model_path=model_path)
