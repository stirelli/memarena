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


class ConsolidatingClient(FakeMem0Client):
    """Models mem0's update-resolution behavior: once a memory exists for a
    user, a further add() CONSOLIDATES into it (an UPDATE event) — the
    memory count stays flat, only the content changes. The real platform
    does this (its extraction pipeline emits ADD/UPDATE/DELETE events), so
    settle-detection must not require the count to increase."""

    def add(self, messages, *, user_id, metadata=None, timestamp=None):
        bucket = self._store.setdefault(user_id, [])
        if bucket:
            bucket[0]["memory"] = f"{bucket[0]['memory']} | {messages[0]['content']}"
            return {"event_id": "evt-update", "status": "PENDING"}
        return super().add(messages, user_id=user_id, metadata=metadata, timestamp=timestamp)


class NoopAddClient(FakeMem0Client):
    """add() is accepted but never changes anything observable (a NOOP
    event, or a lost write) — the poll must eventually give up loudly."""

    def add(self, messages, *, user_id, metadata=None, timestamp=None):
        return {"event_id": "evt-noop", "status": "PENDING"}


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


class TestSettleDetection:
    """add() returns once the write is observable (time-to-settled).
    Review finding F4: settling must be detected as ANY change to the
    namespace's memories, not only a count increase — mem0's own update
    resolution keeps the count flat on consolidation."""

    def _fast_poll(self, monkeypatch):
        import memarena.providers.mem0_adapter as adapter_module
        monkeypatch.setattr(adapter_module, "POLL_TIMEOUT_S", 0.3)
        monkeypatch.setattr(adapter_module, "POLL_INTERVAL_S", 0.01)

    def test_consolidating_add_settles_without_count_increase(self, monkeypatch):
        self._fast_poll(monkeypatch)
        provider = Mem0Provider({"top_k": 5}, client=ConsolidatingClient())
        provider.reset("ns1")
        provider.add("ns1", [{"role": "user", "content": "I live in Berlin."}],
                      session_id="s1", timestamp="2026-01-01T00:00:00Z")
        # Second add consolidates (count stays 1). It must settle, not
        # stall until the poll deadline and surface as a false infra error.
        provider.add("ns1", [{"role": "user", "content": "I moved to Amsterdam."}],
                      session_id="s2", timestamp="2026-02-01T00:00:00Z")
        results = provider.search("ns1", "Amsterdam", top_k=5)
        assert len(results) == 1
        assert "Amsterdam" in results[0].content

    def test_noop_add_times_out_loudly(self, monkeypatch):
        self._fast_poll(monkeypatch)
        provider = Mem0Provider({"top_k": 5}, client=NoopAddClient())
        provider.reset("ns1")
        with pytest.raises(ProviderError):
            provider.add("ns1", [{"role": "user", "content": "hi"}],
                          session_id="s1", timestamp="2026-01-01T00:00:00Z")
