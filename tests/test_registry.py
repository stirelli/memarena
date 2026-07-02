import pytest

from memarena.datasets.longmemeval_v2 import LongMemEvalV2Loader
from memarena.datasets.smoke import SmokeDatasetLoader
from memarena.providers.baseline_rag import BaselineRAGProvider
from memarena.registry import get_dataset_class, get_provider_class


def test_get_provider_class_known():
    assert get_provider_class("baseline_rag") is BaselineRAGProvider


def test_get_provider_class_unknown_raises_with_helpful_message():
    with pytest.raises(KeyError, match="baseline_rag"):
        get_provider_class("nonexistent")


def test_get_dataset_class_known():
    assert get_dataset_class("smoke") is SmokeDatasetLoader


def test_get_dataset_class_longmemeval_v2_known():
    assert get_dataset_class("longmemeval_v2") is LongMemEvalV2Loader


def test_get_dataset_class_unknown_raises_with_helpful_message():
    with pytest.raises(KeyError, match="smoke"):
        get_dataset_class("nonexistent")
