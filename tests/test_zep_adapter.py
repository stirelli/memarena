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

    def test_memory_representation_follows_search_scope(self):
        # Episodes return raw transcript chunks (verbatim metrics apply);
        # edges return distilled facts (verbatim metrics are N/A).
        assert _provider().memory_representation == "extractive"
        edges = ZepProvider({"search_scope": "edges"}, client=FakeZep())
        assert edges.memory_representation == "abstractive"


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


class FakeGraphiti:
    """Async fake for graphiti-core's surface the self-hosted mode touches."""

    def __init__(self):
        self.episodes: list[dict] = []
        self.removed: list[str] = []
        self.search_calls: list[dict] = []
        self.search_results = None  # set per test

    async def add_episode(self, *, name, episode_body, source, source_description, reference_time, group_id):
        uuid = f"ep-{len(self.episodes) + 1}"
        self.episodes.append({"uuid": uuid, "name": name, "body": episode_body,
                              "reference_time": reference_time, "group_id": group_id})
        return SimpleNamespace(episode=SimpleNamespace(uuid=uuid))

    async def remove_episode(self, uuid):
        self.removed.append(uuid)

    async def search_(self, query, *, config, group_ids):
        self.search_calls.append({"query": query, "limit": config.limit, "group_ids": group_ids})
        return self.search_results or SimpleNamespace(episodes=[], episode_reranker_scores=[])


def _oss_provider(fake_graphiti=None):
    return ZepProvider(
        {"top_k": 5, "self_hosted": True}, graphiti=fake_graphiti or FakeGraphiti(),
    )


class TestSelfHostedGraphiti:
    def test_add_is_synchronous_and_settle_is_free(self):
        fake = FakeGraphiti()
        provider = _oss_provider(fake)
        provider.reset("ns1")
        provider.add("ns1", [{"role": "user", "content": "hello"}],
                     session_id="s1", timestamp="2023-05-20T02:21:00Z")
        provider.settle("ns1")  # must not raise and must not need polling

        [episode] = fake.episodes
        assert episode["group_id"] == "ns1"
        assert "user: hello" in episode["body"]
        assert episode["reference_time"].year == 2023

    def test_long_session_splits_into_multiple_episodes(self):
        fake = FakeGraphiti()
        provider = _oss_provider(fake)
        provider.reset("ns1")
        provider.add("ns1", [{"role": "user", "content": "x" * (MAX_EPISODE_CHARS * 2)}],
                     session_id="s1", timestamp="2023-01-01T00:00:00Z")
        assert len(fake.episodes) == 3
        assert "".join(e["body"] for e in fake.episodes).count("x" * 100) > 0

    def test_re_reset_removes_only_that_namespaces_episodes(self):
        fake = FakeGraphiti()
        provider = _oss_provider(fake)
        provider.reset("ns1")
        provider.reset("ns2")
        provider.add("ns1", [{"role": "user", "content": "a"}], session_id="s1",
                     timestamp="2023-01-01T00:00:00Z")
        provider.add("ns2", [{"role": "user", "content": "b"}], session_id="s2",
                     timestamp="2023-01-01T00:00:00Z")
        provider.reset("ns1")
        assert fake.removed == ["ep-1"]

    def test_search_maps_episodes_and_scopes_to_namespace(self):
        fake = FakeGraphiti()
        from datetime import datetime
        fake.search_results = SimpleNamespace(
            episodes=[SimpleNamespace(uuid="ep-9", content="the fact", name="s1-0",
                                       valid_at=datetime(2023, 1, 1))],
            episode_reranker_scores=[0.7],
        )
        provider = _oss_provider(fake)
        provider.reset("ns1")
        [record] = provider.search("ns1", "what fact?", top_k=5)
        assert record.id == "ep-9"
        assert record.content == "the fact"
        assert record.score == 0.7
        assert record.created_at.startswith("2023-01-01")
        assert fake.search_calls == [{"query": "what fact?", "limit": 5, "group_ids": ["ns1"]}]

    def test_info_reports_graphiti_version_and_per_token(self):
        info = _oss_provider().info()
        assert info.client_version == "graphiti-core-0.29.2"
        assert info.pricing_model == "per_token"

    def test_graphiti_failure_becomes_provider_error(self):
        fake = FakeGraphiti()

        async def boom(**kwargs):
            raise RuntimeError("kuzu exploded")

        fake.add_episode = boom
        provider = _oss_provider(fake)
        provider.reset("ns1")
        with pytest.raises(ProviderError, match="graphiti"):
            provider.add("ns1", [{"role": "user", "content": "x"}],
                         session_id="s1", timestamp="2023-01-01T00:00:00Z")
