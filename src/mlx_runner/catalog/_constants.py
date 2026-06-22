from __future__ import annotations

from typing import Tuple

from .catalog_model import CatalogModel

_B = 1_000_000_000

# Smallest to largest. All are 4-bit instruct builds from the mlx-community org.
CATALOG: Tuple[CatalogModel, ...] = (
    CatalogModel("Llama 3.2 1B Instruct", "mlx-community/Llama-3.2-1B-Instruct-4bit", int(1.24 * _B)),
    CatalogModel("Qwen2.5 1.5B Instruct", "mlx-community/Qwen2.5-1.5B-Instruct-4bit", int(1.54 * _B)),
    CatalogModel("Llama 3.2 3B Instruct", "mlx-community/Llama-3.2-3B-Instruct-4bit", int(3.21 * _B)),
    CatalogModel("Qwen2.5 7B Instruct", "mlx-community/Qwen2.5-7B-Instruct-4bit", int(7.62 * _B)),
    CatalogModel("Llama 3.1 8B Instruct", "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit", int(8.03 * _B)),
    CatalogModel("Qwen2.5 14B Instruct", "mlx-community/Qwen2.5-14B-Instruct-4bit", int(14.8 * _B)),
    CatalogModel("Qwen2.5 32B Instruct", "mlx-community/Qwen2.5-32B-Instruct-4bit", int(32.5 * _B)),
    CatalogModel("Qwen2.5 72B Instruct", "mlx-community/Qwen2.5-72B-Instruct-4bit", int(72.7 * _B)),
)
