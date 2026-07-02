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
        assert info.pricing_model == "per_request"  # platform API mode — see LICENSES.md
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
    """§8 Day 3: mem0 is accept+settle (see the adapter docstring for the
    measured rationale). add() must return fast without polling; settle()
    waits for a non-empty, quiescent namespace snapshot — ANY change resets
    the quiet window (review finding F4's count-increase trap still
    applies: consolidation keeps the count flat), and an all-no-op
    ingestion times out loudly instead of faking success."""

    def _fast_poll(self, monkeypatch):
        import memarena.providers.mem0_adapter as adapter_module
        monkeypatch.setattr(adapter_module, "POLL_INTERVAL_S", 0.001)
        monkeypatch.setattr(adapter_module, "QUIET_WINDOW_S", 0.005)
        monkeypatch.setattr(adapter_module, "SETTLE_TIMEOUT_S", 0.25)

    def test_add_is_accept_only_and_never_polls(self):
        # Even a client whose writes never become observable must not make
        # add() block or raise — acceptance is the only contract here.
        provider = Mem0Provider({"top_k": 5}, client=NoopAddClient())
        provider.reset("ns1")
        provider.add("ns1", [{"role": "user", "content": "hi"}],
                     session_id="s1", timestamp="2026-01-01T00:00:00Z")

    def test_settle_accepts_quiescent_consolidated_state(self, monkeypatch):
        self._fast_poll(monkeypatch)
        provider = Mem0Provider({"top_k": 5}, client=ConsolidatingClient())
        provider.reset("ns1")
        provider.add("ns1", [{"role": "user", "content": "I live in Berlin."}],
                     session_id="s1", timestamp="2026-01-01T00:00:00Z")
        # Consolidation keeps the count at 1; settle must still pass once
        # the (changed) snapshot is stable.
        provider.add("ns1", [{"role": "user", "content": "I moved to Amsterdam."}],
                     session_id="s2", timestamp="2026-02-01T00:00:00Z")
        provider.settle("ns1")
        results = provider.search("ns1", "Amsterdam", top_k=5)
        assert len(results) == 1
        assert "Amsterdam" in results[0].content

    def test_settle_waits_out_a_still_changing_snapshot(self, monkeypatch):
        self._fast_poll(monkeypatch)

        class TrickleClient(FakeMem0Client):
            """Memories appear one poll at a time — settle must not return
            while the snapshot is still moving."""

            def __init__(self):
                super().__init__()
                self.trickle = ["fact A", "fact B", "fact C"]
                self.polls = 0

            def get_all(self, *, filters):
                self.polls += 1
                served = self.trickle[: self.polls]
                return {"results": [{"id": f"m{i}", "memory": m, "updated_at": None}
                                     for i, m in enumerate(served)]}

        client = TrickleClient()
        provider = Mem0Provider({"top_k": 5}, client=client)
        provider.reset("ns1")
        provider.add("ns1", [{"role": "user", "content": "x"}],
                     session_id="s1", timestamp="2026-01-01T00:00:00Z")
        provider.settle("ns1")
        assert client.polls > len(client.trickle)  # kept polling until quiet

    def test_all_noop_ingestion_times_out_loudly(self, monkeypatch):
        self._fast_poll(monkeypatch)
        provider = Mem0Provider({"top_k": 5}, client=NoopAddClient())
        provider.reset("ns1")
        provider.add("ns1", [{"role": "user", "content": "hi"}],
                     session_id="s1", timestamp="2026-01-01T00:00:00Z")
        with pytest.raises(ProviderError, match="did not settle"):
            provider.settle("ns1")

    def test_oss_mode_add_is_synchronous_and_settle_free(self):
        """self_hosted mode: add() passes NO timestamp param (OSS gates it
        behind a platform key), never tracks pending writes, and settle()
        therefore never polls."""
        class OssFake(FakeMem0Client):
            def __init__(self):
                super().__init__()
                self.get_all_calls = 0
                self.add_kwargs: list[dict] = []

            def add(self, messages, *, user_id, metadata=None, timestamp=None):
                self.add_kwargs.append({"metadata": metadata, "timestamp": timestamp})
                return super().add(messages, user_id=user_id, metadata=metadata)

            def get_all(self, *, filters):
                self.get_all_calls += 1
                return super().get_all(filters=filters)

        client = OssFake()
        provider = Mem0Provider({"top_k": 5, "self_hosted": True}, client=client)
        provider.reset("ns1")
        provider.add("ns1", [{"role": "user", "content": "My dog's name is Biscuit."}],
                     session_id="s1", timestamp="2026-01-01T00:00:00Z")
        provider.settle("ns1")

        assert client.add_kwargs[0]["timestamp"] is None
        assert client.add_kwargs[0]["metadata"]["source_timestamp"] == "2026-01-01T00:00:00Z"
        assert client.get_all_calls == 0
        assert provider.supports_temporal is False
        assert provider.info().pricing_model == "per_token"
        [record] = provider.search("ns1", "dog", top_k=5)
        assert "Biscuit" in record.content

    def test_settle_without_pending_adds_is_a_noop(self):
        class CountingClient(FakeMem0Client):
            def __init__(self):
                super().__init__()
                self.get_all_calls = 0

            def get_all(self, *, filters):
                self.get_all_calls += 1
                return super().get_all(filters=filters)

        client = CountingClient()
        provider = Mem0Provider({"top_k": 5}, client=client)
        provider.reset("ns1")
        provider.settle("ns1")  # returns immediately — no adds pending
        assert client.get_all_calls == 0
