from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime
from typing import Protocol

from mem0.exceptions import MemoryError as Mem0MemoryError
from mem0.exceptions import RateLimitError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from memarena.errors import ProviderError
from memarena.providers.base import MemoryProvider, MemoryRecord, ProviderInfo

CLIENT_VERSION = "2.0.11"  # pinned — mem0ai==2.0.11 in pyproject.toml
POLL_INTERVAL_S = 10.0
# The namespace snapshot must hold still this long (with at least one write
# observed) before ingestion counts as settled. Measured live 2026-07-02 on
# a real 8-session LongMemEval item: extractions run concurrently server-side,
# memories trickle in until ~190s after accept, and mid-pipeline gaps of up
# to 26s were observed between snapshot changes — the window must exceed
# the largest expected gap or settle fires early.
QUIET_WINDOW_S = 50.0
SETTLE_TIMEOUT_S = 600.0


class Mem0ClientProtocol(Protocol):
    def add(self, messages, *, user_id: str, metadata: dict | None = None, timestamp: int | None = None) -> dict: ...
    def get_all(self, *, filters: dict) -> dict: ...
    def search(self, query: str, *, filters: dict, top_k: int = 5) -> dict: ...
    def delete_all(self, *, user_id: str) -> dict: ...


def _iso_to_unix(timestamp: str) -> int:
    return int(datetime.fromisoformat(timestamp.replace("Z", "+00:00")).timestamp())


def _default_client(api_key: str | None) -> Mem0ClientProtocol:
    from mem0 import MemoryClient
    key = api_key or os.environ.get("MEM0_API_KEY")
    if not key:
        raise ProviderError("MEM0_API_KEY is not set; mem0 adapter needs it (set it in .env, never hardcode it).")
    return MemoryClient(api_key=key)


def _oss_client(config: dict):
    """mem0 OSS (self-hosted) client: the vendor's open-source extraction
    pipeline running locally against an embedded qdrant store. Extraction
    LLM is pinned in the provider config (part of the config digest)."""
    if not os.environ.get("OPENAI_API_KEY"):
        raise ProviderError("OPENAI_API_KEY is not set; mem0 self-hosted mode runs its extraction LLM through it.")
    from mem0 import Memory
    return Memory.from_config({
        "llm": {"provider": "openai", "config": {
            "model": config.get("oss_llm_model", "gpt-4.1-mini"), "temperature": 0.0,
        }},
        "vector_store": {"provider": "qdrant", "config": {
            "path": config.get("oss_vector_store_path", ".cache/mem0_oss"),
        }},
    })


def _wrap_mem0_errors(fn):
    def wrapped(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Mem0MemoryError as exc:
            raise ProviderError(f"mem0 client error [{exc.error_code}]: {exc.message}") from exc
    return wrapped


_retry_rate_limit = retry(
    retry=retry_if_exception_type(RateLimitError),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, max=4),
    reraise=True,
)


class Mem0Provider(MemoryProvider):
    """mem0 adapter (§5.5). Namespace = mem0 user_id (spec's convention).

    TWO MODES, selected by config `self_hosted` (see
    configs/providers/mem0.default.yaml for why OSS is the Day 3 default):
    - self_hosted: true — mem0 OSS, the vendor's open-source extraction
      pipeline running locally (pinned extraction LLM, embedded qdrant).
      add() extracts synchronously, so add_latency_ms is TIME-TO-SETTLED
      and settle() is a no-op. No platform quota; cost = metered OpenAI
      usage (pricing_model per_token).
    - self_hosted absent/false — the mem0 platform API. Measured live
      2026-07-02: the free tier bills search() AND get_all() against ONE
      1,000/month retrieval bucket ({"event_type": "SEARCH"} quota errors),
      which the Day 3 run exhausted 21 items in; the evidence journal is
      preserved at results/day3-v1-four-providers/
      mem0__platform_quota_blocked__journal.jsonl.

    PLATFORM LATENCY SEMANTICS (§8 Day 3; contrast letta/baseline, compare zep):
    mem0's write path (/v3/memories/add/) is asynchronous — add() returns
    {"event_id", "status": "PENDING"} and extraction happens server-side.
    Day 2 made add() poll get_all() per call (time-to-settled); live V1
    measurement (2026-07-02) showed one ~11k-char session takes >30s to
    extract while accepts take ~0.7s and sessions of an item are processed
    concurrently — per-add polling would serialize those extractions and
    multiply ingestion wall-clock by sessions-per-item. Day 3 therefore
    moves mem0 to the accept+settle contract (providers/base.py): add() is
    TIME-TO-ACCEPTED, and settle() carries the pipeline time, journaled
    separately as settle_latency_ms.

    settle() condition, stated honestly: the namespace snapshot (ids,
    contents, updated_at) must show at least one change since reset AND
    hold still for QUIET_WINDOW_S. Quiescence is a heuristic — mem0
    exposes no per-event status in mem0ai==2.0.11 (the add response's
    event_id has no queryable endpoint in this SDK), so a server-side
    stall longer than the quiet window could declare settle early; the
    window is sized from live measurements and the residue shows up as
    (deterministically journaled) retrieval misses, never hidden.

    Known limitation (kept from Day 2, now per item instead of per
    session): an item whose EVERY session extracts to a true no-op leaves
    the snapshot empty, so settle() polls to the deadline and raises;
    the runner records a visible infra_error, never a fake success.
    """

    supports_temporal = True  # platform accepts a timestamp per add(); False in self-hosted mode (see __init__)
    supports_update_resolution = True  # mem0's own extraction resolves updates

    def __init__(self, config: dict, *, client: Mem0ClientProtocol | None = None):
        self._config = config
        self._top_k_default = config.get("top_k", 5)
        self._self_hosted = bool(config.get("self_hosted"))
        if client is not None:
            self._client = client
        else:
            self._client = _oss_client(config) if self._self_hosted else _default_client(config.get("api_key"))
        self._pending_adds: dict[str, int] = {}
        if self._self_hosted:
            # OSS gates the timestamp-backdating parameter behind a platform
            # API key (measured live 2026-07-02: ValueError "Temporal
            # reasoning requires a Mem0 API key"), so the flag is honest.
            self.supports_temporal = False

    def info(self) -> ProviderInfo:
        digest = hashlib.sha256(json.dumps(self._config, sort_keys=True).encode()).hexdigest()
        return ProviderInfo(
            name="mem0", client_version=CLIENT_VERSION, config_digest=digest,
            pricing_model="per_token" if self._self_hosted else "per_request",
        )

    @_wrap_mem0_errors
    def reset(self, namespace: str) -> None:
        self._client.delete_all(user_id=namespace)
        self._pending_adds[namespace] = 0

    @_retry_rate_limit
    def _add_with_retry(self, messages, *, user_id: str, metadata: dict, timestamp: int) -> dict:
        return self._client.add(messages, user_id=user_id, metadata=metadata, timestamp=timestamp)

    @_retry_rate_limit
    def _search_with_retry(self, query: str, *, filters: dict, top_k: int) -> dict:
        return self._client.search(query, filters=filters, top_k=top_k)

    @_wrap_mem0_errors
    def add(self, namespace: str, messages: list[dict[str, str]], *, session_id: str, timestamp: str) -> None:
        metadata = {"session_id": session_id, "source_timestamp": timestamp}
        if self._self_hosted:
            # Synchronous local extraction: settled on return (time-to-settled
            # semantics, like baseline/letta). No timestamp param — OSS gates
            # it behind a platform API key.
            self._client.add(messages, user_id=namespace, metadata=metadata)
            return
        self._add_with_retry(
            messages, user_id=namespace, metadata=metadata, timestamp=_iso_to_unix(timestamp),
        )
        self._pending_adds[namespace] = self._pending_adds.get(namespace, 0) + 1

    def _memories_snapshot(self, namespace: str) -> frozenset[tuple]:
        results = self._client.get_all(filters={"user_id": namespace}).get("results", [])
        return frozenset((r.get("id"), r.get("memory"), r.get("updated_at")) for r in results)

    @_wrap_mem0_errors
    def settle(self, namespace: str) -> None:
        if self._pending_adds.get(namespace, 0) == 0:
            return
        deadline = time.monotonic() + SETTLE_TIMEOUT_S
        last_snapshot = self._memories_snapshot(namespace)
        last_change = time.monotonic()
        while time.monotonic() < deadline:
            quiet_for = time.monotonic() - last_change
            if last_snapshot and quiet_for >= QUIET_WINDOW_S:
                self._pending_adds[namespace] = 0
                return
            time.sleep(POLL_INTERVAL_S)
            snapshot = self._memories_snapshot(namespace)
            if snapshot != last_snapshot:
                last_snapshot, last_change = snapshot, time.monotonic()
        raise ProviderError(
            f"mem0 ingestion did not settle within {SETTLE_TIMEOUT_S}s for namespace={namespace!r} "
            f"({self._pending_adds.get(namespace, 0)} accepted adds, no stable non-empty snapshot)"
        )

    @_wrap_mem0_errors
    def search(self, namespace: str, query: str, *, top_k: int = 5) -> list[MemoryRecord]:
        response = self._search_with_retry(query, filters={"user_id": namespace}, top_k=top_k)
        records = []
        for r in response.get("results", []):
            records.append(MemoryRecord(
                id=r["id"], content=r["memory"], metadata=r.get("metadata") or {},
                score=r.get("score"), created_at=r.get("created_at"),
            ))
        return records
