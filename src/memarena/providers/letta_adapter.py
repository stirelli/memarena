from __future__ import annotations

import hashlib
import json
import os

from letta_client import APIStatusError, LettaError, NotFoundError

from memarena.errors import ProviderError
from memarena.providers.base import MemoryProvider, MemoryRecord, ProviderInfo

CLIENT_VERSION = "1.12.1"  # pinned — letta-client==1.12.1 in pyproject.toml
SHARED_AGENT_NAME = "memarena-shared-store"


def _default_client(api_key: str | None):
    from letta_client import Letta
    key = api_key or os.environ.get("LETTA_API_KEY")
    if not key:
        raise ProviderError("LETTA_API_KEY is not set; letta adapter needs it (set it in .env, never hardcode it).")
    # The Stainless-generated client already retries 429/5xx with backoff
    # (max_retries=2 default), so no tenacity layer here.
    return Letta(api_key=key)


def _wrap_letta_errors(fn):
    def wrapped(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except APIStatusError as exc:
            raise ProviderError(f"letta api error [{exc.status_code}]: {exc}") from exc
        except LettaError as exc:
            raise ProviderError(f"letta client error: {exc}") from exc
    return wrapped


def namespace_tag(namespace: str) -> str:
    return f"ns:{namespace}"


def session_to_passage_text(messages: list[dict[str, str]], *, session_id: str, timestamp: str) -> str:
    """One archival passage per dataset session: a plain transcript with the
    session's original timestamp in the header. Deterministic, lossless —
    evidence turns survive byte-for-byte (content matching, §5.4)."""
    header = f"Conversation session {session_id} at {timestamp}:"
    lines = [f"{m['role']}: {m['content']}" for m in messages]
    return "\n".join([header, *lines])


class LettaProvider(MemoryProvider):
    """Letta adapter (§5.5, §8 Day 3). Store/recall via archival memory:
    each dataset session is one archival passage (agents.passages.create),
    retrieval via archival semantic search (agents.passages.search). A full
    conversational MemGPT loop is deliberately NOT driven — Level-1
    measures the memory store, not the agent policy.

    SHARED-AGENT DESIGN, forced and verified live (2026-07-02): Letta
    Cloud's free tier hard-caps the account at 3 agents (API 402,
    {"limit": 3}), so one-agent-per-namespace cannot hold a 200-item run.
    All namespaces therefore live on ONE shared agent and are isolated by
    passage tags: writes tag each passage `ns:{namespace}`, searches
    filter on that tag. Isolation was verified live in both directions
    (a namespace's search never returns another namespace's passages, and
    an unknown namespace returns nothing). Vendors are invited to PR an
    agent-per-namespace config for paid tiers (§6.2).

    reset() semantics under this design: the FIRST reset of a process
    deletes and recreates the shared agent (a whole-store wipe — correct
    for the runner, which starts every run with fresh namespaces);
    subsequent resets delete the passages this process inserted for that
    namespace (tracked by id). A namespace never touched by this process
    is empty by construction after the initial wipe.

    LATENCY SEMANTICS (goal item 4): agents.passages.create embeds and
    indexes synchronously — the passage is queryable when the call
    returns — so the runner's add_latency_ms for letta is TIME-TO-SETTLED
    (same quantity as baseline_rag; contrast zep/mem0). settle() keeps the
    no-op default.

    Agent creation uses Letta Cloud's server-side defaults for model and
    embedding (quickstart behavior); overrides can be pinned in
    configs/providers/letta.default.yaml (`model`, `embedding`) and become
    part of the config digest."""

    supports_temporal = True  # passages carry created_at; search returns timestamps
    supports_update_resolution = False  # archival passages are append-only

    def __init__(self, config: dict, *, client=None):
        self._config = config
        self._client = client or _default_client(config.get("api_key"))
        self._agent_id: str | None = None
        self._store_wiped = False
        self._passage_ids: dict[str, list[str]] = {}

    def info(self) -> ProviderInfo:
        digest = hashlib.sha256(json.dumps(self._config, sort_keys=True).encode()).hexdigest()
        return ProviderInfo(
            name="letta", client_version=CLIENT_VERSION, config_digest=digest, pricing_model="per_request",
        )

    def _agent(self) -> str:
        if self._agent_id:
            return self._agent_id
        for agent in self._client.agents.list(name=SHARED_AGENT_NAME):
            self._agent_id = agent.id
            return agent.id
        raise ProviderError(f"letta shared agent {SHARED_AGENT_NAME!r} does not exist; reset() must run first")

    @_wrap_letta_errors
    def reset(self, namespace: str) -> None:
        if not self._store_wiped:
            for agent in self._client.agents.list(name=SHARED_AGENT_NAME):
                try:
                    self._client.agents.delete(agent.id)
                except NotFoundError:
                    pass  # already gone — idempotent wipe
            create_kwargs: dict = {"name": SHARED_AGENT_NAME}
            for key in ("model", "embedding"):
                if key in self._config:
                    create_kwargs[key] = self._config[key]
            self._agent_id = self._client.agents.create(**create_kwargs).id
            self._store_wiped = True
            self._passage_ids.clear()
            return
        for passage_id in self._passage_ids.pop(namespace, []):
            try:
                self._client.agents.passages.delete(passage_id, agent_id=self._agent())
            except NotFoundError:
                pass
        # Any namespace this process never wrote to is empty by construction
        # after the initial whole-store wipe.

    @_wrap_letta_errors
    def add(self, namespace: str, messages: list[dict[str, str]], *, session_id: str, timestamp: str) -> None:
        text = session_to_passage_text(messages, session_id=session_id, timestamp=timestamp)
        created = self._client.agents.passages.create(
            self._agent(), text=text, created_at=timestamp,
            tags=[namespace_tag(namespace), f"session:{session_id}"],
        )
        self._passage_ids.setdefault(namespace, []).extend(p.id for p in created)

    @_wrap_letta_errors
    def search(self, namespace: str, query: str, *, top_k: int = 5) -> list[MemoryRecord]:
        response = self._client.agents.passages.search(
            self._agent(), query=query, top_k=top_k,
            tags=[namespace_tag(namespace)], tag_match_mode="any",
        )
        records = []
        for result in response.results or []:
            records.append(MemoryRecord(
                id=result.id, content=result.content, metadata={"tags": result.tags or []},
                score=None,  # letta's passage search returns no relevance score
                created_at=result.timestamp,
            ))
        return records[:top_k]
