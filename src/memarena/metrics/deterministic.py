from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher

import numpy as np

DEFAULT_K_VALUES: tuple[int, ...] = (1, 3, 5, 10)


def normalize_content(text: str) -> str:
    return " ".join(text.strip().lower().split())


def content_matches(a: str, b: str, *, fuzzy_threshold: float = 0.85) -> bool:
    """Gold-evidence matching (§5.4): exact match after normalization, with a
    normalized fuzzy fallback for minor punctuation/whitespace drift."""
    na, nb = normalize_content(a), normalize_content(b)
    if na == nb:
        return True
    return SequenceMatcher(None, na, nb).ratio() >= fuzzy_threshold


def recall_at_k(retrieved_contents: list[str], gold_evidence: list[str], k: int) -> float | None:
    """1.0 if any gold evidence content-matches a top-k retrieved record, else
    0.0. None when the item has no gold evidence (e.g. abstention items —
    not applicable to Level-1 retrieval metrics)."""
    if not gold_evidence:
        return None
    top = retrieved_contents[:k]
    hit = any(content_matches(gold, retrieved) for gold in gold_evidence for retrieved in top)
    return 1.0 if hit else 0.0


def reciprocal_rank(retrieved_contents: list[str], gold_evidence: list[str]) -> float | None:
    if not gold_evidence:
        return None
    for rank, retrieved in enumerate(retrieved_contents, start=1):
        if any(content_matches(gold, retrieved) for gold in gold_evidence):
            return 1.0 / rank
    return 0.0


def mean_of_defined(values: list[float | None]) -> float | None:
    defined = [v for v in values if v is not None]
    if not defined:
        return None
    return sum(defined) / len(defined)


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(values, p))


def percentile_of_defined(values: list[float | None], p: float) -> float | None:
    defined = [v for v in values if v is not None]
    if not defined:
        return None
    return percentile(defined, p)


@dataclass(frozen=True)
class ItemMetric:
    item_id: str
    recall_at_k: dict[int, float | None]
    reciprocal_rank: float | None
    add_latency_ms: float | None
    search_latency_ms: float


def compute_item_metric(
    item_id: str,
    retrieved_contents: list[str],
    gold_evidence: list[str],
    add_latency_ms: float | None,
    search_latency_ms: float,
    *,
    k_values: tuple[int, ...] = DEFAULT_K_VALUES,
) -> ItemMetric:
    return ItemMetric(
        item_id=item_id,
        recall_at_k={k: recall_at_k(retrieved_contents, gold_evidence, k) for k in k_values},
        reciprocal_rank=reciprocal_rank(retrieved_contents, gold_evidence),
        add_latency_ms=add_latency_ms,
        search_latency_ms=search_latency_ms,
    )


@dataclass(frozen=True)
class RunMetrics:
    recall_at_k: dict[int, float | None]
    mrr: float | None
    add_latency_p50_ms: float | None
    add_latency_p95_ms: float | None
    search_latency_p50_ms: float | None  # None for an empty run — never a fabricated 0.0
    search_latency_p95_ms: float | None
    n_items: int
    n_scored_items: int  # items with gold evidence — excludes pure-abstention items


def aggregate_run(items: list[ItemMetric], *, k_values: tuple[int, ...] = DEFAULT_K_VALUES) -> RunMetrics:
    add_latencies = [item.add_latency_ms for item in items]
    search_latencies: list[float | None] = [item.search_latency_ms for item in items]
    return RunMetrics(
        recall_at_k={k: mean_of_defined([item.recall_at_k[k] for item in items]) for k in k_values},
        mrr=mean_of_defined([item.reciprocal_rank for item in items]),
        add_latency_p50_ms=percentile_of_defined(add_latencies, 50),
        add_latency_p95_ms=percentile_of_defined(add_latencies, 95),
        search_latency_p50_ms=percentile_of_defined(search_latencies, 50),
        search_latency_p95_ms=percentile_of_defined(search_latencies, 95),
        n_items=len(items),
        n_scored_items=sum(1 for item in items if item.reciprocal_rank is not None),
    )
