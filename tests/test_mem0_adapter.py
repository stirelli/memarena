import pytest

from memarena.errors import ProviderError
from memarena.providers.mem0_adapter import Mem0Provider


class FakeMem0Client:
    """Deterministic stand-in for mem0.MemoryClient — models the real
    async-add-then-poll behavior (add() lands after get_all() reflects it)
    without any network or sleep."""

    def __init__(self):
        self.deleted: list[str] = []
        self._store: dict[str, list[dict]] = {}
        self.add_calls: list[dict] = []
        self._next_id = 0

    def add(self, messages, *, user_id, metadata=None, timestamp=None):
        self.add_calls.append({"messages": messages, "user_id": user_id})
        bucket = self._store.setdefault(user_id, [])
        for m in messages:
            self._next_id += 1
            bucket.append({"id": f"mem-{self._next_id}", "memory": m["content"], "score": None})
        return {"event_id": "evt-1", "status": "PENDING"}

    def get_all(self, *, filters):
        user_id = filters["user_id"]
        return {"results": self._store.get(user_id, [])}

    def search(self, query, *, filters, top_k=5):
        user_id = filters["user_id"]
        matches = [m for m in self._store.get(user_id, []) if query.lower() in m["memory"].lower()]
        for i, m in enumerate(matches):
            m["score"] = 1.0 - i * 0.1
        return {"results": matches[:top_k]}

    def delete_all(self, *, user_id):
        self.deleted.append(user_id)
        self._store[user_id] = []
        return {"message": "ok"}


class FailingAddClient(FakeMem0Client):
    def add(self, messages, *, user_id, metadata=None, timestamp=None):
        from mem0.exceptions import RateLimitError
        raise RateLimitError(message="slow down", error_code="RATE_001")


class TestMem0Provider:
    def setup_method(self):
        self.client = FakeMem0Client()
        self.provider = Mem0Provider({"top_k": 5}, client=self.client)

    def test_info(self):
        info = self.provider.info()
        assert info.name == "mem0"
        assert info.pricing_model == "self_hosted"  # free-tier hosted API — see LICENSES.md
        assert len(info.config_digest) == 64

    def test_reset_calls_delete_all(self):
        self.provider.reset("ns1")
        assert self.client.deleted == ["ns1"]

    def test_add_then_search_returns_relevant_record(self):
        self.provider.reset("ns1")
        self.provider.add("ns1", [{"role": "user", "content": "My dog's name is Biscuit."}],
                           session_id="s1", timestamp="2026-01-01T00:00:00Z")
        results = self.provider.search("ns1", "dog", top_k=5)
        assert len(results) == 1
        assert "Biscuit" in results[0].content
        assert results[0].score is not None

    def test_add_passes_user_id_and_messages_through(self):
        self.provider.reset("ns1")
        self.provider.add("ns1", [{"role": "user", "content": "hi"}], session_id="s1", timestamp="2026-01-01T00:00:00Z")
        assert self.client.add_calls[0]["user_id"] == "ns1"

    def test_search_top_k_is_respected(self):
        self.provider.reset("ns1")
        for i in range(10):
            self.provider.add("ns1", [{"role": "user", "content": f"fact number {i}"}],
                               session_id=f"s{i}", timestamp="2026-01-01T00:00:00Z")
        results = self.provider.search("ns1", "fact", top_k=3)
        assert len(results) == 3

    def test_namespaces_are_isolated(self):
        self.provider.reset("ns1")
        self.provider.reset("ns2")
        self.provider.add("ns1", [{"role": "user", "content": "secret A"}],
                           session_id="s1", timestamp="2026-01-01T00:00:00Z")
        assert self.provider.search("ns2", "secret", top_k=5) == []

    def test_client_error_raises_provider_error(self):
        provider = Mem0Provider({"top_k": 5}, client=FailingAddClient())
        provider.reset("ns1")
        with pytest.raises(ProviderError):
            provider.add("ns1", [{"role": "user", "content": "hi"}], session_id="s1", timestamp="2026-01-01T00:00:00Z")
