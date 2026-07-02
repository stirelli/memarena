from __future__ import annotations

import hashlib
import json
import os
import time

from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential
from zep_cloud.core.api_error import ApiError
from zep_cloud.errors import NotFoundError

from memarena.errors import ProviderError
from memarena.providers.base import MemoryProvider, MemoryRecord, ProviderInfo

CLIENT_VERSION = "3.23.0"  # pinned — zep-cloud==3.23.0 in pyproject.toml

# One text episode per <= MAX_EPISODE_CHARS chunk of a session transcript.
# Sized under Zep's ~10k-char graph.add ceiling; long sessions are SPLIT,
# never truncated — evidence content must survive ingestion byte-for-byte.
MAX_EPISODE_CHARS = 9000
# Zep's search endpoint caps query length; questions in our datasets are far
# shorter, but the guard keeps an oversized query a loud error, not a 400.
MAX_QUERY_CHARS = 400
SETTLE_POLL_INTERVAL_S = 10.0
SETTLE_TIMEOUT_S = 1800.0  # free tier: ~10-27s per episode, lower-priority queue (measured 2026-07-02)
EPISODES_LASTN_MARGIN = 50  # fetch a little beyond our own count, defensive
# Client-side pacing: the free plan enforces VARIABLE rate limits and a
# sustained 429 storm was measured live (2026-07-02) once un-paced retries
# started hammering. Every API call waits at least this long after the
# previous one.
MIN_REQUEST_INTERVAL_S = 1.0


def _default_client(api_key: str | None):
    from zep_cloud.client import Zep
    key = api_key or os.environ.get("ZEP_API_KEY")
    if not key:
        raise ProviderError("ZEP_API_KEY is not set; zep adapter needs it (set it in .env, never hardcode it).")
    return Zep(api_key=key)


def _is_retryable(exc: BaseException) -> bool:
    return isinstance(exc, ApiError) and getattr(exc, "status_code", None) in (429, 500, 502, 503, 504)


# Patient by design: a transient free-plan throttle must stall the run, not
# poison the journal with infra_error cascades (measured live 2026-07-02:
# an impatient retry converted a throttling episode into ~150 failed items).
_retry_transient = retry(
    retry=retry_if_exception(_is_retryable),
    stop=stop_after_attempt(8),
    wait=wait_exponential(multiplier=2, max=120),
    reraise=True,
)


def _wrap_zep_errors(fn):
    def wrapped(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except ApiError as exc:
            status = getattr(exc, "status_code", "?")
            raise ProviderError(f"zep api error [{status}]: {getattr(exc, 'body', exc)}") from exc
    return wrapped


def split_content(content: str, *, max_chars: int = MAX_EPISODE_CHARS) -> list[str]:
    if not content:
        return [content]
    return [content[i: i + max_chars] for i in range(0, len(content), max_chars)]


def session_to_transcript(messages: list[dict[str, str]], *, session_id: str, timestamp: str) -> str:
    header = f"Conversation session {session_id} at {timestamp}:"
    return "\n".join([header, *(f"{m['role']}: {m['content']}" for m in messages)])


class ZepProvider(MemoryProvider):
    """Zep Cloud adapter (§5.5, §8 Day 3). Namespace = Zep user_id; every
    dataset session becomes text episodes on that user's graph via
    graph.add (one episode per <=9k-char transcript chunk), with
    `created_at` carrying the session's original timestamp so Zep's
    temporal graph sees historical time, not ingestion time.

    INGESTION PATH, decided against live measurements (2026-07-02): Zep's
    chat quickstart (thread.add_messages, one message per turn) bills one
    flex credit PER MESSAGE and the free tier processes episodes at
    ~10-27s each on a lower-priority queue — per-turn ingestion of a
    200-item LongMemEval V1 sample is ~16.5k credits against a 10k/month
    plan and >100h of pipeline time. graph.add text episodes are Zep's
    other first-class ingestion surface (their "adding data to the graph"
    path), cost ~14 credits/item here, and carry identical content. The
    thread-based config remains the vendor-invited PR for paid runs
    (§6.2); this default is documented in configs/providers/zep.default.yaml.

    LATENCY SEMANTICS (goal item 4; contrast with mem0_adapter.py Day 2,
    compare Day 3): graph.add returns once the episode is ACCEPTED; graph
    construction (entity extraction, fact invalidation) continues
    server-side. add_latency_ms for zep is therefore TIME-TO-ACCEPTED.
    The remaining pipeline time is measured by settle(): it polls
    graph.episode.get_by_user_id until every episode this adapter added is
    visible AND processed, and the runner journals that separately as
    settle_latency_ms. Accepted and settled are DIFFERENT quantities;
    reports must present them with the settle column, never as one number.

    Search uses graph.search with scope="episodes" (raw ingested content) as
    the Level-1 retrieval surface: content-based gold matching (§5.4) needs
    source text, and Zep's distilled edges/facts are exercised by the
    Level-2 answer-correctness track instead (Day 4). scope is config-
    overridable (`search_scope`) and vendors can PR their preferred config
    (§6.2)."""

    supports_temporal = True  # graph.add created_at carries original timestamps into the temporal graph
    supports_update_resolution = True  # Zep's graph invalidates superseded facts

    def __init__(self, config: dict, *, client=None):
        self._config = config
        self._search_scope = config.get("search_scope", "episodes")
        if self._search_scope not in ("episodes", "edges"):
            raise ProviderError(f"unsupported zep search_scope {self._search_scope!r} (episodes|edges)")
        self._client = client or _default_client(config.get("api_key"))
        self._episodes_added: dict[str, int] = {}
        self._last_request_at = 0.0

    def _pace(self) -> None:
        """Client-side rate limiting (MIN_REQUEST_INTERVAL_S) applied before
        every API call — see the constant's comment for the live evidence."""
        wait = self._last_request_at + MIN_REQUEST_INTERVAL_S - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        self._last_request_at = time.monotonic()

    def info(self) -> ProviderInfo:
        digest = hashlib.sha256(json.dumps(self._config, sort_keys=True).encode()).hexdigest()
        return ProviderInfo(
            name="zep", client_version=CLIENT_VERSION, config_digest=digest, pricing_model="per_request",
        )

    @_wrap_zep_errors
    def reset(self, namespace: str) -> None:
        try:
            self._pace()
            self._client.user.delete(namespace)
        except NotFoundError:
            pass  # idempotent wipe: a user that never existed is already reset
        self._pace()
        self._client.user.add(user_id=namespace)
        self._episodes_added[namespace] = 0

    @_retry_transient
    def _graph_add_with_retry(self, *, user_id: str, data: str, created_at: str) -> None:
        self._pace()
        self._client.graph.add(user_id=user_id, type="text", data=data, created_at=created_at)

    @_wrap_zep_errors
    def add(self, namespace: str, messages: list[dict[str, str]], *, session_id: str, timestamp: str) -> None:
        transcript = session_to_transcript(messages, session_id=session_id, timestamp=timestamp)
        for chunk in split_content(transcript):
            self._graph_add_with_retry(user_id=namespace, data=chunk, created_at=timestamp)
            self._episodes_added[namespace] = self._episodes_added.get(namespace, 0) + 1

    @_wrap_zep_errors
    def settle(self, namespace: str) -> None:
        expected = self._episodes_added.get(namespace, 0)
        if expected == 0:
            return
        deadline = time.monotonic() + SETTLE_TIMEOUT_S
        lastn = expected + EPISODES_LASTN_MARGIN
        while time.monotonic() < deadline:
            try:
                self._pace()
                episodes = self._client.graph.episode.get_by_user_id(namespace, lastn=lastn).episodes or []
                if len(episodes) >= expected and all(e.processed for e in episodes):
                    return
            except ApiError as exc:
                # A throttled poll is "no data yet", not a failed item — the
                # deadline still bounds the wait (live 2026-07-02 evidence).
                if not _is_retryable(exc):
                    raise
            time.sleep(SETTLE_POLL_INTERVAL_S)
        raise ProviderError(
            f"zep ingestion did not settle within {SETTLE_TIMEOUT_S}s for namespace={namespace!r} "
            f"(expected >= {expected} processed episodes)"
        )

    @_retry_transient
    def _search_with_retry(self, *, query: str, user_id: str, top_k: int):
        self._pace()
        return self._client.graph.search(
            query=query, user_id=user_id, scope=self._search_scope, limit=top_k,
        )

    @_wrap_zep_errors
    def search(self, namespace: str, query: str, *, top_k: int = 5) -> list[MemoryRecord]:
        if len(query) > MAX_QUERY_CHARS:
            raise ProviderError(
                f"query exceeds zep's {MAX_QUERY_CHARS}-char search limit ({len(query)} chars); "
                "refusing to silently truncate a benchmark query"
            )
        results = self._search_with_retry(query=query, user_id=namespace, top_k=top_k)
        records = []
        if self._search_scope == "episodes":
            for episode in results.episodes or []:
                records.append(MemoryRecord(
                    id=episode.uuid_, content=episode.content or "",
                    metadata={"thread_id": episode.thread_id},
                    score=episode.score, created_at=episode.created_at,
                ))
        else:  # edges
            for edge in results.edges or []:
                records.append(MemoryRecord(
                    id=edge.uuid_, content=edge.fact or "", metadata={},
                    score=getattr(edge, "score", None), created_at=getattr(edge, "created_at", None),
                ))
        return records[:top_k]
