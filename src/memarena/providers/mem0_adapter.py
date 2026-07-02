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
POLL_INTERVAL_S = 1.0
POLL_TIMEOUT_S = 30.0  # empirically ~5s to settle (confirmed live 2026-07-01); generous margin


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

    Mem0's write path (/v3/memories/add/) is asynchronous — add() returns
    {"event_id", "status": "PENDING"} immediately server-side (confirmed
    live 2026-07-01: a memory took ~5s to become visible via get_all()). To
    honor the MemoryProvider sync-façade contract (Appendix A) and to make
    an immediately-following search() reliable, add() polls get_all() for
    this namespace until the observed memory count increases, up to
    POLL_TIMEOUT_S. This means add_latency_ms genuinely includes multi-second
    settle time — that's real, not a measurement artifact.
    """

    supports_temporal = True  # mem0 accepts a timestamp per add() call
    supports_update_resolution = True  # mem0's own extraction resolves updates

    def __init__(self, config: dict, *, client: Mem0ClientProtocol | None = None):
        self._config = config
        self._top_k_default = config.get("top_k", 5)
        self._client = client or _default_client(config.get("api_key"))

    def info(self) -> ProviderInfo:
        digest = hashlib.sha256(json.dumps(self._config, sort_keys=True).encode()).hexdigest()
        return ProviderInfo(
            name="mem0", client_version=CLIENT_VERSION, config_digest=digest, pricing_model="self_hosted",
        )

    @_wrap_mem0_errors
    def reset(self, namespace: str) -> None:
        self._client.delete_all(user_id=namespace)

    @_retry_rate_limit
    def _add_with_retry(self, messages, *, user_id: str, metadata: dict, timestamp: int) -> dict:
        return self._client.add(messages, user_id=user_id, metadata=metadata, timestamp=timestamp)

    @_retry_rate_limit
    def _search_with_retry(self, query: str, *, filters: dict, top_k: int) -> dict:
        return self._client.search(query, filters=filters, top_k=top_k)

    @_wrap_mem0_errors
    def add(self, namespace: str, messages: list[dict[str, str]], *, session_id: str, timestamp: str) -> None:
        before = len(self._client.get_all(filters={"user_id": namespace}).get("results", []))
        self._add_with_retry(
            messages, user_id=namespace,
            metadata={"session_id": session_id, "source_timestamp": timestamp},
            timestamp=_iso_to_unix(timestamp),
        )
        self._poll_until_visible(namespace, before)

    def _poll_until_visible(self, namespace: str, before_count: int) -> None:
        deadline = time.monotonic() + POLL_TIMEOUT_S
        while time.monotonic() < deadline:
            after = len(self._client.get_all(filters={"user_id": namespace}).get("results", []))
            if after > before_count:
                return
            time.sleep(POLL_INTERVAL_S)
        raise ProviderError(f"mem0 add() did not become visible within {POLL_TIMEOUT_S}s for namespace={namespace!r}")

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
