from __future__ import annotations

import inspect
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
        # TypeError. Pass it only when the installed signature accepts it (named or
        # via **kwargs); otherwise thread it through tokenizer_config, and omit it
        # entirely when False so a standard load still works on older mlx-lm.
        tok_cfg = dict(tokenizer_config or {})
        params = inspect.signature(mlx_lm.load).parameters
        accepts_trc = "trust_remote_code" in params or any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
        )
        load_kwargs = {"adapter_path": adapter_path, "lazy": lazy}
        if accepts_trc:
            load_kwargs["trust_remote_code"] = trust_remote_code
        elif trust_remote_code:
            tok_cfg["trust_remote_code"] = True
        load_kwargs["tokenizer_config"] = tok_cfg
        model, tokenizer = mlx_lm.load(model_path, **load_kwargs)
        return cls(model, tokenizer, model_path=model_path)
