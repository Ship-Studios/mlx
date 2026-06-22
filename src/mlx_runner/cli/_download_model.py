from __future__ import annotations


def _download_model(repo_id: str) -> str:
    """Fetch a model snapshot into the local HF cache; return its path.

    ``huggingface_hub`` ships with mlx-lm; import it lazily so the rest of the
    CLI works without it.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise RuntimeError(
            "huggingface_hub is not installed. Install it with `pip install mlx-lm` "
            "(it is pulled in as a dependency)."
        ) from e
    return snapshot_download(repo_id)
