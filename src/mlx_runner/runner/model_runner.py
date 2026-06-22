from __future__ import annotations

from ._model_runner_init import _ModelRunnerInitMixin
from ._model_runner_load import _ModelRunnerLoadMixin
from ._model_runner_make_prompt_cache import _ModelRunnerMakePromptCacheMixin
from ._model_runner_load_prompt_cache import _ModelRunnerLoadPromptCacheMixin
from ._model_runner_save_prompt_cache import _ModelRunnerSavePromptCacheMixin
from ._model_runner_build_and_save_prompt_cache import (
    _ModelRunnerBuildAndSavePromptCacheMixin,
)
from ._model_runner_format_prompt import _ModelRunnerFormatPromptMixin
from ._model_runner_build_kwargs import _ModelRunnerBuildKwargsMixin
from ._model_runner_apply_seed import _ModelRunnerApplySeedMixin
from ._model_runner_stream import _ModelRunnerStreamMixin
from ._model_runner_generate import _ModelRunnerGenerateMixin


class ModelRunner(
    _ModelRunnerInitMixin,
    _ModelRunnerLoadMixin,
    _ModelRunnerMakePromptCacheMixin,
    _ModelRunnerLoadPromptCacheMixin,
    _ModelRunnerSavePromptCacheMixin,
    _ModelRunnerBuildAndSavePromptCacheMixin,
    _ModelRunnerFormatPromptMixin,
    _ModelRunnerBuildKwargsMixin,
    _ModelRunnerApplySeedMixin,
    _ModelRunnerStreamMixin,
    _ModelRunnerGenerateMixin,
):
    """Loads an mlx-lm model once and serves repeated generations from it.

    The model is loaded once during initialization (or via :meth:`load`),
    and subsequent generations reuse the same model instance. This is more
    efficient than loading a new model for each generation.

    The class is assembled from one mixin per method (one method per file);
    see the ``_model_runner_*`` modules in this subpackage.

    Attributes:
        model: The loaded model object.
        tokenizer: The tokenizer object.
        model_path: Path to the model (for display purposes).
        last_stats: Statistics from the last generation (if any).
    """

    pass
