from __future__ import annotations

from memarena.datasets.base import DatasetLoader
from memarena.datasets.longmemeval_v1 import LongMemEvalV1Loader
from memarena.datasets.longmemeval_v2 import LongMemEvalV2Loader
from memarena.datasets.smoke import SmokeDatasetLoader
from memarena.providers.base import MemoryProvider
from memarena.providers.baseline_rag import BaselineRAGProvider
from memarena.providers.letta_adapter import LettaProvider
from memarena.providers.mem0_adapter import Mem0Provider
from memarena.providers.zep_adapter import ZepProvider

PROVIDER_REGISTRY: dict[str, type[MemoryProvider]] = {
    "baseline_rag": BaselineRAGProvider,
    "mem0": Mem0Provider,
    "zep": ZepProvider,
    "letta": LettaProvider,
}

DATASET_REGISTRY: dict[str, type[DatasetLoader]] = {
    "smoke": SmokeDatasetLoader,
    "longmemeval_v1": LongMemEvalV1Loader,
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
