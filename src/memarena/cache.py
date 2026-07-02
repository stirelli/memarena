from __future__ import annotations

from memarena.providers.base import ProviderInfo


def ingestion_cache_key(info: ProviderInfo, *, dataset_digest: str, namespace: str) -> str:
    """Cache key for 'has this namespace already been ingested this run'
    (§5.3). Changing provider config, client version, or dataset revision
    must invalidate the cache — all four are part of the key."""
    return f"{info.name}:{info.client_version}:{info.config_digest}:{dataset_digest}:{namespace}"


class IngestionCache:
    """In-run (non-persistent) ingestion cache. Tracks which (provider,
    config, dataset, namespace) combinations have already had reset()+add()
    performed this run, so items sharing a namespace (e.g. LongMemEval-V2's
    per-domain haystacks) pay ingestion cost once, not once per item.

    Scope: in-process only, does not survive across separate CLI
    invocations. A persistent sqlite-backed cache (spec §5.2) is deferred —
    persisting ingestion state across runs is only safe for providers with
    real external storage (mem0) and actively unsafe for in-memory ones
    (baseline_rag, which starts every process with an empty store).
    """

    def __init__(self) -> None:
        self._seen: set[str] = set()

    def already_ingested(self, key: str) -> bool:
        return key in self._seen

    def mark_ingested(self, key: str) -> None:
        self._seen.add(key)
