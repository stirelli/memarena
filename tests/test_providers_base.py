import pytest

from memarena.errors import ProviderError
from memarena.providers.base import MemoryProvider, MemoryRecord, ProviderInfo


def test_memory_provider_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        MemoryProvider()


def test_concrete_provider_must_implement_all_abstract_methods():
    class IncompleteProvider(MemoryProvider):
        def info(self) -> ProviderInfo:
            return ProviderInfo(
                name="incomplete", client_version="0.0.1",
                config_digest="deadbeef", pricing_model="self_hosted",
            )

    with pytest.raises(TypeError):
        IncompleteProvider()


def test_full_concrete_provider_can_be_instantiated_and_used():
    class FakeProvider(MemoryProvider):
        def __init__(self):
            self.store: dict[str, list[MemoryRecord]] = {}

        def info(self) -> ProviderInfo:
            return ProviderInfo(
                name="fake", client_version="0.0.1",
                config_digest="deadbeef", pricing_model="self_hosted",
            )

        def reset(self, namespace: str) -> None:
            self.store[namespace] = []

        def add(self, namespace, messages, *, session_id, timestamp) -> None:
            self.store.setdefault(namespace, []).append(
                MemoryRecord(id=session_id, content=messages[0]["content"])
            )

        def search(self, namespace, query, *, top_k=5):
            return self.store.get(namespace, [])[:top_k]

    provider = FakeProvider()
    provider.reset("ns")
    provider.add("ns", [{"role": "user", "content": "hi"}], session_id="s1", timestamp="2026-01-01T00:00:00Z")
    results = provider.search("ns", "hi", top_k=5)

    assert len(results) == 1
    assert results[0].content == "hi"
    assert provider.supports_temporal is False
    assert provider.supports_update_resolution is False


def test_provider_info_is_frozen():
    info = ProviderInfo(
        name="fake", client_version="0.0.1",
        config_digest="deadbeef", pricing_model="self_hosted",
    )
    with pytest.raises(AttributeError):
        info.name = "other"


def test_memory_record_defaults():
    record = MemoryRecord(id="r1", content="content")
    assert record.metadata == {}
    assert record.score is None
    assert record.created_at is None


def test_provider_error_is_an_exception():
    with pytest.raises(ProviderError):
        raise ProviderError("boom")
