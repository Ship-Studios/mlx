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
        model, tokenizer = mlx_lm.load(
            model_path,
            adapter_path=adapter_path,
            tokenizer_config=tokenizer_config or {},
            trust_remote_code=trust_remote_code,
            lazy=lazy,
        )
        return cls(model, tokenizer, model_path=model_path)
