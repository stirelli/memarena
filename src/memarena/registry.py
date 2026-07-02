from __future__ import annotations

from memarena.datasets.base import DatasetLoader
from memarena.datasets.longmemeval_v2 import LongMemEvalV2Loader
from memarena.datasets.smoke import SmokeDatasetLoader
from memarena.providers.base import MemoryProvider
from memarena.providers.baseline_rag import BaselineRAGProvider

PROVIDER_REGISTRY: dict[str, type[MemoryProvider]] = {
    "baseline_rag": BaselineRAGProvider,
}

DATASET_REGISTRY: dict[str, type[DatasetLoader]] = {
    "smoke": SmokeDatasetLoader,
    "longmemeval_v2": LongMemEvalV2Loader,
}


def get_provider_class(name: str) -> type[MemoryProvider]:
    try:
        return PROVIDER_REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"Unknown provider adapter '{name}'. Known adapters: {sorted(PROVIDER_REGISTRY)}"
        ) from None


def get_dataset_class(name: str) -> type[DatasetLoader]:
    try:
        return DATASET_REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"Unknown dataset '{name}'. Known datasets: {sorted(DATASET_REGISTRY)}"
        ) from None
