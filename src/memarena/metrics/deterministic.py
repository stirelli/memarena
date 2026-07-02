from __future__ import annotations

import math
from dataclasses import dataclass
from difflib import SequenceMatcher

import numpy as np

DEFAULT_K_VALUES: tuple[int, ...] = (1, 3, 5, 10)


def normalize_content(text: str) -> str:
    return " ".join(text.strip().lower().split())


MIN_CONTAINMENT_CHARS = 40  # containment fallback guard: tiny fragments (tail chunks, "yes") never count as evidence


def content_matches(a: str, b: str, *, fuzzy_threshold: float = 0.85) -> bool:
    """Gold-evidence matching (§5.4): exact match after normalization, a
    containment fallback (either normalized string contained in the other,
    provided the contained one is >= MIN_CONTAINMENT_CHARS — this is what
    lets a fixed-size chunk of an evidence turn, or a whole session that
    embeds the evidence turn, count as a hit), and a normalized fuzzy
    fallback for minor punctuation/whitespace drift. Providers that store
    rewritten/distilled memories instead of source text can fail all three;
    that is a documented property of content-based gold mapping, annotated
    on the leaderboard, never silently corrected."""
    na, nb = normalize_content(a), normalize_content(b)
    if na == nb:
        return True
    shorter, longer = (na, nb) if len(na) <= len(nb) else (nb, na)
    if len(shorter) >= MIN_CONTAINMENT_CHARS and shorter in longer:
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


def ndcg_at_k(retrieved_contents: list[str], gold_evidence: list[str], k: int) -> float | None:
    """NDCG@k with binary relevance — the LongMemEval paper's official
    retrieval metric. Gains are gold-consuming: each distinct gold evidence
    item credits at most ONE retrieved record (the earliest-ranked match),
    so duplicate retrievals of the same evidence cannot push DCG above the
    ideal. IDCG places the min(k, #distinct gold) relevant records at the
    top ranks. None when the item has no gold evidence (same exclusion rule
    as Recall@k / RR)."""
    if not gold_evidence:
        return None
    distinct_gold: list[str] = []
    for gold in gold_evidence:
        if not any(content_matches(gold, seen) for seen in distinct_gold):
            distinct_gold.append(gold)

    remaining = list(distinct_gold)
    dcg = 0.0
    for rank, retrieved in enumerate(retrieved_contents[:k], start=1):
        matched = next((i for i, gold in enumerate(remaining) if content_matches(gold, retrieved)), None)
        if matched is not None:
            remaining.pop(matched)
            dcg += 1.0 / math.log2(rank + 1)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(k, len(distinct_gold)) + 1))
    return dcg / idcg


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
    ndcg_at_k: dict[int, float | None]
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
        ndcg_at_k={k: ndcg_at_k(retrieved_contents, gold_evidence, k) for k in k_values},
        reciprocal_rank=reciprocal_rank(retrieved_contents, gold_evidence),
        add_latency_ms=add_latency_ms,
        search_latency_ms=search_latency_ms,
    )


@dataclass(frozen=True)
class RunMetrics:
    recall_at_k: dict[int, float | None]
    ndcg_at_k: dict[int, float | None]
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
        ndcg_at_k={k: mean_of_defined([item.ndcg_at_k[k] for item in items]) for k in k_values},
        mrr=mean_of_defined([item.reciprocal_rank for item in items]),
        add_latency_p50_ms=percentile_of_defined(add_latencies, 50),
        add_latency_p95_ms=percentile_of_defined(add_latencies, 95),
        search_latency_p50_ms=percentile_of_defined(search_latencies, 50),
        search_latency_p95_ms=percentile_of_defined(search_latencies, 95),
        n_items=len(items),
        n_scored_items=sum(1 for item in items if item.reciprocal_rank is not None),
    )
