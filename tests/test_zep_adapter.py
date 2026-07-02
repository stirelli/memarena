"""Zep adapter tests (§8 Day 3): fake client, no network. The fake mirrors
the zep-cloud==3.23.0 surface the adapter touches (user, graph.add,
graph.search, graph.episode.get_by_user_id)."""

from types import SimpleNamespace

import pytest
from zep_cloud.core.api_error import ApiError
from zep_cloud.errors import NotFoundError

from memarena.errors import ProviderError
from memarena.providers.zep_adapter import (
    MAX_EPISODE_CHARS,
    MAX_QUERY_CHARS,
    ZepProvider,
    session_to_transcript,
    split_content,
)


class _Episode(SimpleNamespace):
    pass


class FakeZep:
    """Records every call; episode processing is scripted per test via
    `episode_batches` (each settle poll consumes one batch)."""

    def __init__(self):
        self.users: set[str] = set()
        self.deleted_users: list[str] = []
        self.graph_adds: list[dict] = []
        self.episode_batches: list[list[_Episode]] = []
        self.episode_polls = 0
        self.search_calls: list[dict] = []
        self.search_episodes: list[_Episode] = []

        fake = self

        class _User:
            def add(self, *, user_id):
                fake.users.add(user_id)

            def delete(self, user_id):
                if user_id not in fake.users:
                    raise NotFoundError(body=None)
                fake.users.discard(user_id)
                fake.deleted_users.append(user_id)

        class _EpisodeApi:
            def get_by_user_id(self, user_id, *, lastn):
                fake.episode_polls += 1
                batch = fake.episode_batches.pop(0) if fake.episode_batches else []
                return SimpleNamespace(episodes=batch)

        class _Graph:
            episode = _EpisodeApi()

            def add(self, *, user_id, type, data, created_at=None):
                fake.graph_adds.append(
                    {"user_id": user_id, "type": type, "data": data, "created_at": created_at})
                return _Episode(uuid_=f"ep-{len(fake.graph_adds)}", processed=False)

            def search(self, *, query, user_id, scope, limit):
                fake.search_calls.append({"query": query, "user_id": user_id, "scope": scope, "limit": limit})
                return SimpleNamespace(episodes=fake.search_episodes, edges=None)

        self.user = _User()
        self.graph = _Graph()


@pytest.fixture(autouse=True)
def _no_pacing(monkeypatch):
    """Client-side pacing is real-time behavior; unit tests run unpaced."""
    monkeypatch.setattr("memarena.providers.zep_adapter.MIN_REQUEST_INTERVAL_S", 0.0)


def _provider(fake=None, config=None):
    return ZepProvider(config or {"top_k": 5}, client=fake or FakeZep())


def _processed(n):
    return [_Episode(processed=True) for _ in range(n)]


class TestReset:
    def test_reset_on_missing_user_is_idempotent(self):
        fake = FakeZep()
        _provider(fake).reset("ns1")
        assert "ns1" in fake.users

    def test_reset_deletes_then_recreates(self):
        fake = FakeZep()
        provider = _provider(fake)
        provider.reset("ns1")
        provider.reset("ns1")
        assert fake.deleted_users == ["ns1"]
        assert "ns1" in fake.users


class TestAdd:
    def test_one_text_episode_per_session_with_backdated_created_at(self):
        fake = FakeZep()
        provider = _provider(fake)
        provider.reset("ns1")
        provider.add("ns1", [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi!"},
        ], session_id="s1", timestamp="2023-05-20T02:21:00Z")

        [call] = fake.graph_adds
        assert call["user_id"] == "ns1"
        assert call["type"] == "text"
        assert call["created_at"] == "2023-05-20T02:21:00Z"
        assert "user: hello" in call["data"]
        assert "assistant: hi!" in call["data"]
        assert "s1" in call["data"]

    def test_long_session_is_split_never_truncated(self):
        fake = FakeZep()
        provider = _provider(fake)
        provider.reset("ns1")
        long_content = "x" * (MAX_EPISODE_CHARS * 2)
        provider.add("ns1", [{"role": "user", "content": long_content}],
                     session_id="s1", timestamp="2023-01-01T00:00:00Z")

        parts = [c["data"] for c in fake.graph_adds]
        assert len(parts) == 3  # header pushes the transcript past 2 chunks
        assert "".join(parts) == session_to_transcript(
            [{"role": "user", "content": long_content}],
            session_id="s1", timestamp="2023-01-01T00:00:00Z")

    def test_split_content_roundtrips(self):
        content = "abc" * 9000
        assert "".join(split_content(content)) == content
        assert all(len(p) <= MAX_EPISODE_CHARS for p in split_content(content))


class TestSettle:
    def test_settle_waits_for_all_episodes_processed(self, monkeypatch):
        monkeypatch.setattr("memarena.providers.zep_adapter.SETTLE_POLL_INTERVAL_S", 0.0)
        fake = FakeZep()
        provider = _provider(fake)
        provider.reset("ns1")
        provider.add("ns1", [{"role": "user", "content": "a"}],
                     session_id="s1", timestamp="2023-01-01T00:00:00Z")
        provider.add("ns1", [{"role": "user", "content": "b"}],
                     session_id="s2", timestamp="2023-01-02T00:00:00Z")

        fake.episode_batches = [
            [_Episode(processed=True)],                       # not all episodes visible yet
            [_Episode(processed=True), _Episode(processed=False)],  # visible but one unprocessed
            _processed(2),                                    # settled
        ]
        provider.settle("ns1")
        assert fake.episode_polls == 3

    def test_settle_times_out_loudly(self, monkeypatch):
        monkeypatch.setattr("memarena.providers.zep_adapter.SETTLE_POLL_INTERVAL_S", 0.0)
        monkeypatch.setattr("memarena.providers.zep_adapter.SETTLE_TIMEOUT_S", 0.05)
        fake = FakeZep()
        provider = _provider(fake)
        provider.reset("ns1")
        provider.add("ns1", [{"role": "user", "content": "a"}],
                     session_id="s1", timestamp="2023-01-01T00:00:00Z")
        fake.episode_batches = [[_Episode(processed=False)]] * 10_000

        with pytest.raises(ProviderError, match="did not settle"):
            provider.settle("ns1")

    def test_settle_without_adds_is_a_noop(self):
        fake = FakeZep()
        provider = _provider(fake)
        provider.reset("ns1")
        provider.settle("ns1")
        assert fake.episode_polls == 0

    def test_settle_survives_transient_throttling(self, monkeypatch):
        """A 429 during a settle poll is 'no data yet', never a poisoned
        item (live 2026-07-02: an impatient poll converted one throttling
        episode into ~150 infra_errors)."""
        monkeypatch.setattr("memarena.providers.zep_adapter.SETTLE_POLL_INTERVAL_S", 0.0)
        fake = FakeZep()
        provider = _provider(fake)
        provider.reset("ns1")
        provider.add("ns1", [{"role": "user", "content": "a"}],
                     session_id="s1", timestamp="2023-01-01T00:00:00Z")

        calls = {"n": 0}
        real_get = fake.graph.episode.get_by_user_id

        def flaky_get(user_id, *, lastn):
            calls["n"] += 1
            if calls["n"] <= 2:
                raise ApiError(status_code=429, body="Rate limit exceeded for FREE plan")
            return real_get(user_id, lastn=lastn)

        fake.graph.episode.get_by_user_id = flaky_get
        fake.episode_batches = [_processed(1)]
        provider.settle("ns1")
        assert calls["n"] == 3

    def test_settle_raises_on_non_transient_poll_error(self, monkeypatch):
        monkeypatch.setattr("memarena.providers.zep_adapter.SETTLE_POLL_INTERVAL_S", 0.0)
        fake = FakeZep()
        provider = _provider(fake)
        provider.reset("ns1")
        provider.add("ns1", [{"role": "user", "content": "a"}],
                     session_id="s1", timestamp="2023-01-01T00:00:00Z")

        def forbidden(user_id, *, lastn):
            raise ApiError(status_code=403, body="forbidden")

        fake.graph.episode.get_by_user_id = forbidden
        with pytest.raises(ProviderError, match="zep api error"):
            provider.settle("ns1")


class TestSearch:
    def test_maps_episodes_to_memory_records(self):
        fake = FakeZep()
        fake.search_episodes = [
            _Episode(uuid_="e1", content="the fact", thread_id=None,
                     score=0.9, created_at="2023-01-01T00:00:00Z"),
        ]
        records = _provider(fake).search("ns1", "what fact?", top_k=5)
        assert len(records) == 1
        assert records[0].id == "e1"
        assert records[0].content == "the fact"
        assert records[0].score == 0.9
        assert records[0].created_at == "2023-01-01T00:00:00Z"
        assert fake.search_calls == [
            {"query": "what fact?", "user_id": "ns1", "scope": "episodes", "limit": 5},
        ]

    def test_oversized_query_is_refused_not_truncated(self):
        with pytest.raises(ProviderError, match="char search limit"):
            _provider().search("ns1", "q" * (MAX_QUERY_CHARS + 1), top_k=5)

    def test_unsupported_scope_is_rejected_at_init(self):
        with pytest.raises(ProviderError, match="search_scope"):
            ZepProvider({"search_scope": "nodes"}, client=FakeZep())


class TestErrorWrapping:
    def test_api_error_becomes_provider_error(self):
        fake = FakeZep()

        def boom(*, user_id):
            raise ApiError(status_code=500, body="internal")

        fake.user.add = boom
        with pytest.raises(ProviderError, match="zep api error"):
            _provider(fake).reset("ns1")


class TestInfo:
    def test_info_pins_version_and_digests_config(self):
        info = _provider().info()
        assert info.name == "zep"
        assert info.client_version == "3.23.0"
        assert len(info.config_digest) == 64
