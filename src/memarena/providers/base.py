from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MemoryRecord:
    id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    score: float | None = None          # provider-reported relevance, if any
    created_at: str | None = None       # ISO 8601


@dataclass(frozen=True)
class ProviderInfo:
    name: str                # "mem0"
    client_version: str      # pinned, e.g. "0.1.29"
    config_digest: str       # sha256 of the exact config used
    pricing_model: str       # "per_request" | "per_token" | "self_hosted"


class MemoryProvider(ABC):
    """Minimal contract a memory system must satisfy to be benchmarked.

    Rules for adapters:
    - Synchronous facade; wrap async clients internally. add() must return
      only once the write is SETTLED (queryable by search/get), so the
      runner's add-latency means time-to-settled for every provider:
      async backends include their internal settle/poll time, synchronous
      backends settle on return. reset() is excluded from timing.
    - Do NOT time or aggregate internally; the runner measures wall-clock.
    - Raise memarena.errors.ProviderError with context on failures;
      the runner records the item as `infra_error` (excluded from
      accuracy, reported separately).
    """

    #: capability flags — annotate the leaderboard; never silently penalize
    supports_temporal: bool = False
    supports_update_resolution: bool = False

    @abstractmethod
    def info(self) -> ProviderInfo: ...

    @abstractmethod
    def reset(self, namespace: str) -> None:
        """Idempotently delete all memories under `namespace`."""

    @abstractmethod
    def add(self, namespace: str, messages: list[dict[str, str]],
            *, session_id: str, timestamp: str) -> None:
        """Ingest one session chunk.
        messages: [{"role": "user"|"assistant", "content": str}, ...]
        timestamp: ISO 8601 session time (temporal providers use it;
                   others may ignore — flag drives leaderboard note).
        """

    @abstractmethod
    def search(self, namespace: str, query: str,
               *, top_k: int = 5) -> list[MemoryRecord]:
        """Return up to top_k most relevant memories for `query`."""
