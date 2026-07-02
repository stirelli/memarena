from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import threading
import time
from datetime import datetime

from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential
from zep_cloud.core.api_error import ApiError
from zep_cloud.errors import NotFoundError

from memarena.errors import ProviderError
from memarena.providers.base import MemoryProvider, MemoryRecord, ProviderInfo

CLIENT_VERSION = "3.23.0"  # pinned — zep-cloud==3.23.0 in pyproject.toml
GRAPHITI_VERSION = "0.29.2"  # pinned — graphiti-core==0.29.2 in pyproject.toml (self-hosted mode)

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


class _LoopRunner:
    """One persistent event loop on a daemon thread: graphiti's async
    clients (httpx-based) bind to the loop they first run on, so per-call
    asyncio.run() would strand them on closed loops."""

    def __init__(self):
        self._loop = asyncio.new_event_loop()
        threading.Thread(target=self._loop.run_forever, daemon=True).start()

    def run(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()


def _wrap_graphiti_errors(fn):
    def wrapped(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(f"graphiti (zep self-hosted) error: {exc}") from exc
    return wrapped


class ZepProvider(MemoryProvider):
    """Zep adapter (§5.5, §8 Day 3), TWO MODES selected by config
    `self_hosted` (Day 3 default: self-hosted graphiti — see the __init__
    comment and configs/providers/zep.default.yaml for the live evidence
    that forced it; the docstring below describes the CLOUD mode kept for
    paid runs via zep.cloud.yaml).

    Cloud mode: namespace = Zep user_id; every dataset session becomes
    text episodes on that user's graph via graph.add (one episode per
    <=9k-char transcript chunk), with `created_at` carrying the session's
    original timestamp so Zep's temporal graph sees historical time, not
    ingestion time.

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

    supports_temporal = True  # created_at / reference_time carries original timestamps into the temporal graph
    supports_update_resolution = True  # the graph invalidates superseded facts

    def __init__(self, config: dict, *, client=None, graphiti=None):
        self._config = config
        self._search_scope = config.get("search_scope", "episodes")
        if self._search_scope not in ("episodes", "edges"):
            raise ProviderError(f"unsupported zep search_scope {self._search_scope!r} (episodes|edges)")
        self._self_hosted = bool(config.get("self_hosted"))
        self._episodes_added: dict[str, int] = {}
        self._last_request_at = 0.0
        if self._self_hosted:
            # SELF-HOSTED MODE (§8 Day 3, measured live 2026-07-02): Zep
            # Cloud's free plan enforces a small rolling request budget —
            # the Day 3 run was hard-throttled after ~1.9k requests (~3h)
            # and a restarted, paced, patiently-retrying shard got ZERO
            # items through (evidence journals preserved in results/). The
            # zep row therefore runs Graphiti, Zep's open-source core,
            # locally: pinned extraction LLM through OPENAI_API_KEY,
            # embedded kuzu graph. add_episode extracts synchronously, so
            # add_latency_ms is TIME-TO-SETTLED here and settle() is a
            # no-op — the cloud path's accept/settle split does not apply.
            # Episode retrieval is BM25 + RRF (graphiti's episode search).
            self._runner = _LoopRunner()
            self._graphiti = graphiti  # tests inject; production builds lazily in reset()
            self._graphiti_db = config.get("graphiti_db_path", ".cache/zep_graphiti/graph.kuzu")
            self._store_wiped = graphiti is not None
            self._episode_uuids: dict[str, list[str]] = {}
            self._kuzu_db = None  # real kuzu handle, needed for FTS index rebuilds
            self._fts_dirty = False
            self._client = None
        else:
            self._client = client or _default_client(config.get("api_key"))

    def _pace(self) -> None:
        """Client-side rate limiting (MIN_REQUEST_INTERVAL_S) applied before
        every API call — see the constant's comment for the live evidence."""
        wait = self._last_request_at + MIN_REQUEST_INTERVAL_S - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        self._last_request_at = time.monotonic()

    def info(self) -> ProviderInfo:
        digest = hashlib.sha256(json.dumps(self._config, sort_keys=True).encode()).hexdigest()
        if self._self_hosted:
            return ProviderInfo(
                name="zep", client_version=f"graphiti-core-{GRAPHITI_VERSION}",
                config_digest=digest, pricing_model="per_token",
            )
        return ProviderInfo(
            name="zep", client_version=CLIENT_VERSION, config_digest=digest, pricing_model="per_request",
        )

    # --- self-hosted (graphiti) machinery -------------------------------

    def _build_graphiti(self):
        if not os.environ.get("OPENAI_API_KEY"):
            raise ProviderError("OPENAI_API_KEY is not set; zep self-hosted mode runs graphiti through it.")
        from graphiti_core import Graphiti
        from graphiti_core.driver.kuzu_driver import KuzuDriver
        from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
        from graphiti_core.llm_client.config import LLMConfig
        from graphiti_core.llm_client.openai_client import OpenAIClient

        model = self._config.get("oss_llm_model", "gpt-4.1-mini")
        llm_config = LLMConfig(model=model, small_model=model, temperature=0.0)
        driver = KuzuDriver(db=self._graphiti_db)
        # graphiti-core==0.29.2 gap: GraphDriver declares `_database` but
        # KuzuDriver never initializes it, and add_episode(group_id=...)
        # reads it before calling clone() — which is a no-op on kuzu (one
        # embedded store; namespaces isolate via the group_id property on
        # nodes/episodes). Any sentinel value unblocks that read.
        driver._database = "kuzu-main"
        # graphiti-core==0.29.2 omission on kuzu: build_indices_and_constraints
        # is a no-op and setup_schema never creates the FTS indexes its own
        # search queries require, so we run get_fulltext_indices ourselves.
        import kuzu
        from graphiti_core.driver.driver import GraphProvider
        from graphiti_core.graph_queries import get_fulltext_indices
        connection = kuzu.Connection(driver.db)
        connection.execute("INSTALL FTS; LOAD EXTENSION FTS;")
        for query in get_fulltext_indices(GraphProvider.KUZU):
            try:
                connection.execute(query)
            except Exception:  # noqa: BLE001 - index already exists on a reopened store
                pass
        connection.close()
        self._kuzu_db = driver.db

        graphiti = Graphiti(
            graph_driver=driver,
            llm_client=OpenAIClient(config=llm_config),
            embedder=OpenAIEmbedder(config=OpenAIEmbedderConfig(embedding_model="text-embedding-3-small")),
        )
        self._runner.run(graphiti.build_indices_and_constraints())
        return graphiti

    def _rebuild_episode_fts(self) -> None:
        """kuzu FTS indexes are STATIC snapshots (verified live 2026-07-02:
        an episode added after CREATE_FTS_INDEX is invisible to
        QUERY_FTS_INDEX until the index is rebuilt), so search() rebuilds
        the episodic index lazily whenever episodes changed since the last
        rebuild. Repetition searches (no new adds) pay nothing."""
        if self._kuzu_db is None:
            return
        import kuzu
        connection = kuzu.Connection(self._kuzu_db)
        connection.execute("LOAD EXTENSION FTS;")
        try:
            connection.execute("CALL DROP_FTS_INDEX('Episodic', 'episode_content');")
        except Exception:  # noqa: BLE001 - nothing to drop on the first build
            pass
        connection.execute(
            "CALL CREATE_FTS_INDEX('Episodic', 'episode_content', ['content', 'source', 'source_description']);"
        )
        connection.close()

    def _g_reset(self, namespace: str) -> None:
        if not self._store_wiped:
            # Whole-store wipe on the first reset of a run: namespaces are
            # fresh per run, so later resets only clear their own episodes.
            db_parent = os.path.dirname(self._graphiti_db)
            if os.path.isdir(db_parent):
                shutil.rmtree(db_parent)
            os.makedirs(db_parent, exist_ok=True)
            self._graphiti = self._build_graphiti()
            self._store_wiped = True
        if self._graphiti is None:
            self._graphiti = self._build_graphiti()
        removed = self._episode_uuids.pop(namespace, [])
        for uuid in removed:
            self._runner.run(self._graphiti.remove_episode(uuid))
        if removed:
            self._fts_dirty = True
        self._episodes_added[namespace] = 0

    def _g_add(self, namespace: str, messages, *, session_id: str, timestamp: str) -> None:
        from graphiti_core.nodes import EpisodeType

        reference_time = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        transcript = session_to_transcript(messages, session_id=session_id, timestamp=timestamp)
        for i, chunk in enumerate(split_content(transcript)):
            result = self._runner.run(self._graphiti.add_episode(
                name=f"{session_id}-{i}",
                episode_body=chunk,
                source=EpisodeType.message,
                source_description="conversation session",
                reference_time=reference_time,
                group_id=namespace,
            ))
            episode = getattr(result, "episode", None)
            if episode is not None:
                self._episode_uuids.setdefault(namespace, []).append(episode.uuid)
        self._fts_dirty = True

    def _g_search(self, namespace: str, query: str, *, top_k: int) -> list[MemoryRecord]:
        from graphiti_core.search.search_config import (
            EpisodeReranker,
            EpisodeSearchConfig,
            EpisodeSearchMethod,
            SearchConfig,
        )

        if self._fts_dirty:
            self._rebuild_episode_fts()
            self._fts_dirty = False

        config = SearchConfig(
            episode_config=EpisodeSearchConfig(
                search_methods=[EpisodeSearchMethod.bm25],
                reranker=EpisodeReranker.rrf,
            ),
            limit=top_k,
        )
        results = self._runner.run(self._graphiti.search_(query, config=config, group_ids=[namespace]))
        scores = list(results.episode_reranker_scores or [])
        records = []
        for i, episode in enumerate(results.episodes or []):
            valid_at = getattr(episode, "valid_at", None)
            records.append(MemoryRecord(
                id=episode.uuid, content=episode.content or "",
                metadata={"name": episode.name},
                score=scores[i] if i < len(scores) else None,
                created_at=valid_at.isoformat() if valid_at is not None else None,
            ))
        return records[:top_k]

    # --- shared entry points ---------------------------------------------

    @_wrap_zep_errors
    def reset(self, namespace: str) -> None:
        if self._self_hosted:
            _wrap_graphiti_errors(self._g_reset)(namespace)
            return
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
        if self._self_hosted:
            _wrap_graphiti_errors(self._g_add)(namespace, messages, session_id=session_id, timestamp=timestamp)
            return
        transcript = session_to_transcript(messages, session_id=session_id, timestamp=timestamp)
        for chunk in split_content(transcript):
            self._graph_add_with_retry(user_id=namespace, data=chunk, created_at=timestamp)
            self._episodes_added[namespace] = self._episodes_added.get(namespace, 0) + 1

    @_wrap_zep_errors
    def settle(self, namespace: str) -> None:
        if self._self_hosted:
            return  # graphiti extraction is synchronous inside add() — settled on return
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
        if self._self_hosted:
            return _wrap_graphiti_errors(self._g_search)(namespace, query, top_k=top_k)
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
