from __future__ import annotations

from typing import List, Optional


class _ModelRunnerFormatPromptMixin:
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

        Args:
            prompt: User prompt string.
            messages: Chat messages list (alternative to prompt).
            system: System prompt (only used with messages).
            add_generation_prompt: Whether to add a generation prompt.

        Returns:
            The formatted prompt as a string or list of token ids.
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
