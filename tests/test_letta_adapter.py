"""Letta adapter tests (§8 Day 3): fake client, no network. The fake mirrors
the letta-client==1.12.1 surface the adapter touches (agents CRUD +
agents.passages create/search/delete) INCLUDING tag-filtered search — the
shared-agent design's namespace isolation lives or dies on that filter
(verified live 2026-07-02; Letta free tier hard-caps at 3 agents)."""

from types import SimpleNamespace

import httpx
import pytest
from letta_client import APIStatusError

from memarena.errors import ProviderError
from memarena.providers.letta_adapter import (
    SHARED_AGENT_NAME,
    LettaProvider,
    namespace_tag,
    session_to_passage_text,
)


class FakeLetta:
    def __init__(self):
        self.agents_by_id: dict[str, SimpleNamespace] = {}
        self.passages: dict[str, list[dict]] = {}  # agent_id -> passages
        self.search_calls: list[dict] = []
        self.create_kwargs: list[dict] = []
        self._next_id = 0

        fake = self

        class _Passages:
            def create(self, agent_id, *, text, created_at=None, tags=None):
                fake._next_id += 1
                passage = {"id": f"p{fake._next_id}", "text": text,
                           "created_at": created_at, "tags": tags or []}
                fake.passages.setdefault(agent_id, []).append(passage)
                return [SimpleNamespace(id=passage["id"])]

            def search(self, agent_id, *, query, top_k=None, tags=None, tag_match_mode=None):
                fake.search_calls.append({"agent_id": agent_id, "query": query, "top_k": top_k,
                                           "tags": tags, "tag_match_mode": tag_match_mode})
                hits = [
                    SimpleNamespace(id=p["id"], content=p["text"],
                                    timestamp=p["created_at"], tags=p["tags"])
                    for p in fake.passages.get(agent_id, [])
                    if not tags or set(tags) & set(p["tags"])
                ]
                return SimpleNamespace(count=len(hits), results=hits[: top_k or 5])

            def delete(self, memory_id, *, agent_id):
                bucket = fake.passages.get(agent_id, [])
                fake.passages[agent_id] = [p for p in bucket if p["id"] != memory_id]

        class _Agents:
            passages = _Passages()

            def list(self, *, name=None):
                return [a for a in fake.agents_by_id.values() if name is None or a.name == name]

            def create(self, **kwargs):
                fake.create_kwargs.append(kwargs)
                fake._next_id += 1
                agent = SimpleNamespace(id=f"agent-{fake._next_id}", name=kwargs["name"])
                fake.agents_by_id[agent.id] = agent
                return agent

            def delete(self, agent_id):
                del fake.agents_by_id[agent_id]
                fake.passages.pop(agent_id, None)

        self.agents = _Agents()


def _provider(fake=None, config=None):
    return LettaProvider(config or {"top_k": 5}, client=fake or FakeLetta())


def _add(provider, ns, content, session_id="s1", timestamp="2023-05-20T02:21:00Z"):
    provider.add(ns, [{"role": "user", "content": content}],
                 session_id=session_id, timestamp=timestamp)


class TestSharedAgentLifecycle:
    def test_first_reset_wipes_and_recreates_the_shared_agent(self):
        fake = FakeLetta()
        stale = fake.agents.create(name=SHARED_AGENT_NAME)
        fake.agents.passages.create(stale.id, text="stale data", tags=["ns:old"])

        _provider(fake).reset("ns1")

        assert len(fake.agents_by_id) == 1
        [agent] = fake.agents_by_id.values()
        assert agent.id != stale.id
        assert fake.passages.get(agent.id, []) == []

    def test_later_resets_do_not_recreate_the_agent(self):
        fake = FakeLetta()
        provider = _provider(fake)
        provider.reset("ns1")
        agent_id = next(iter(fake.agents_by_id))
        provider.reset("ns2")
        assert list(fake.agents_by_id) == [agent_id]

    def test_re_reset_of_a_namespace_deletes_its_passages_only(self):
        fake = FakeLetta()
        provider = _provider(fake)
        provider.reset("ns1")
        provider.reset("ns2")
        _add(provider, "ns1", "fact for ns1")
        _add(provider, "ns2", "fact for ns2")

        provider.reset("ns1")

        [agent_id] = fake.agents_by_id
        remaining_tags = [t for p in fake.passages[agent_id] for t in p["tags"]]
        assert namespace_tag("ns1") not in remaining_tags
        assert namespace_tag("ns2") in remaining_tags

    def test_model_and_embedding_pins_are_forwarded_when_configured(self):
        fake = FakeLetta()
        _provider(fake, {"model": "openai/gpt-4.1", "embedding": "openai/text-embedding-3-small"}).reset("ns1")
        assert fake.create_kwargs[-1]["model"] == "openai/gpt-4.1"
        assert fake.create_kwargs[-1]["embedding"] == "openai/text-embedding-3-small"

    def test_add_before_any_reset_raises(self):
        with pytest.raises(ProviderError, match="reset"):
            _add(_provider(), "never-reset", "x")

    def test_cold_instance_rediscovers_existing_shared_agent(self):
        fake = FakeLetta()
        warm = _provider(fake)
        warm.reset("ns1")
        _add(warm, "ns1", "the fact")
        cold = LettaProvider({"top_k": 5}, client=fake)  # fresh instance, no reset
        assert len(cold.search("ns1", "fact", top_k=5)) == 1


class TestNamespaceIsolation:
    def test_search_only_sees_its_own_namespace(self):
        fake = FakeLetta()
        provider = _provider(fake)
        provider.reset("nsA")
        provider.reset("nsB")
        _add(provider, "nsA", "dog named Biscuit")
        _add(provider, "nsB", "cat named Whiskers")

        a = provider.search("nsA", "pet name", top_k=5)
        b = provider.search("nsB", "pet name", top_k=5)
        assert len(a) == 1 and "dog named Biscuit" in a[0].content
        assert len(b) == 1 and "cat named Whiskers" in b[0].content
        assert "Whiskers" not in a[0].content and "Biscuit" not in b[0].content

    def test_search_passes_namespace_tag_filter(self):
        fake = FakeLetta()
        provider = _provider(fake)
        provider.reset("ns1")
        provider.search("ns1", "q", top_k=3)
        [call] = fake.search_calls
        assert call["tags"] == [namespace_tag("ns1")]
        assert call["tag_match_mode"] == "any"
        assert call["top_k"] == 3


class TestAdd:
    def test_one_passage_per_session_with_timestamp_and_tags(self):
        fake = FakeLetta()
        provider = _provider(fake)
        provider.reset("ns1")
        provider.add("ns1", [
            {"role": "user", "content": "I adopted a dog named Biscuit."},
            {"role": "assistant", "content": "Lovely!"},
        ], session_id="s7", timestamp="2023-05-20T02:21:00Z")

        [agent_id] = fake.agents_by_id
        [passage] = fake.passages[agent_id]
        assert passage["created_at"] == "2023-05-20T02:21:00Z"
        assert passage["tags"] == [namespace_tag("ns1"), "session:s7"]
        assert "I adopted a dog named Biscuit." in passage["text"]

    def test_passage_text_is_lossless_transcript(self):
        text = session_to_passage_text(
            [{"role": "user", "content": "line one"}, {"role": "assistant", "content": "line two"}],
            session_id="s1", timestamp="2023-01-01T00:00:00Z",
        )
        assert text == (
            "Conversation session s1 at 2023-01-01T00:00:00Z:\n"
            "user: line one\n"
            "assistant: line two"
        )


class TestSearchMapping:
    def test_maps_results_to_memory_records(self):
        fake = FakeLetta()
        provider = _provider(fake)
        provider.reset("ns1")
        _add(provider, "ns1", "the fact")
        [record] = provider.search("ns1", "fact", top_k=3)
        assert record.content.endswith("user: the fact")
        assert record.score is None
        assert record.created_at == "2023-05-20T02:21:00Z"
        assert namespace_tag("ns1") in record.metadata["tags"]


class TestErrorWrapping:
    def test_api_status_error_becomes_provider_error(self):
        fake = FakeLetta()
        provider = _provider(fake)
        provider.reset("ns1")

        def boom(agent_id, *, query, top_k=None, tags=None, tag_match_mode=None):
            response = httpx.Response(status_code=429, request=httpx.Request("GET", "https://api.letta.com"))
            raise APIStatusError("rate limited", response=response, body=None)

        fake.agents.passages.search = boom
        with pytest.raises(ProviderError, match="letta api error"):
            provider.search("ns1", "q", top_k=1)


class TestInfo:
    def test_info_pins_version_and_digests_config(self):
        info = _provider().info()
        assert info.name == "letta"
        assert info.client_version == "1.12.1"
        assert len(info.config_digest) == 64
