from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

from memarena.cache import IngestionCache, ingestion_cache_key
from memarena.datasets.base import QAItem
from memarena.errors import ProviderError
from memarena.metrics.deterministic import (
    DEFAULT_K_VALUES,
    ItemMetric,
    RunMetrics,
    aggregate_run,
    compute_item_metric,
)
from memarena.providers.base import MemoryProvider

CHARS_PER_TOKEN = 4  # rough approximation, documented (configs/pricing.yaml)


def estimate_cost_usd(char_count: int, *, usd_per_1k_tokens: float) -> float:
    tokens = char_count / CHARS_PER_TOKEN
    return (tokens / 1000) * usd_per_1k_tokens


@dataclass(frozen=True)
class RunResult:
    run_id: str
    seed: int
    metrics: RunMetrics
    total_cost_usd: float
    budget_truncated: bool
    infra_error_count: int
    n_items_attempted: int


def _add_sessions(provider: MemoryProvider, item: QAItem) -> int:
    """Add every session (the caller resets OUTSIDE the timed window — §5.7:
    add latency is wall-clock around add() only). Returns total ingested
    chars. For async-write providers the adapter's sync façade returns only
    once the write is settled (queryable), so this measures time-to-settled
    consistently across providers."""
    ingest_chars = 0
    for session in item.sessions:
        provider.add(item.namespace, session.messages, session_id=session.session_id, timestamp=session.timestamp)
        ingest_chars += sum(len(m["content"]) for m in session.messages)
    return ingest_chars


def _search(provider: MemoryProvider, item: QAItem, *, top_k: int):
    search_start = time.perf_counter()
    records = provider.search(item.namespace, item.question, top_k=top_k)
    search_latency_ms = (time.perf_counter() - search_start) * 1000
    return records, search_latency_ms


def run(
    provider: MemoryProvider,
    items: list[QAItem],
    *,
    run_id: str,
    seed: int,
    dataset_digest: str,
    repetitions: int = 1,
    top_k: int = 5,
    budget_usd_max: float | None = None,
    pricing: dict | None = None,
    journal_path: str | Path,
    ingestion_cache: IngestionCache | None = None,
    fresh_ingest: bool = False,
) -> RunResult:
    """Runner (§5.3, §8 Day 1+2): seeded item order, JSONL journal, budget
    guard, Level-1 deterministic metrics, and an ingestion cache so items
    that share a namespace (e.g. LongMemEval-V2's per-domain haystacks) pay
    ingestion cost once, not once per item. `fresh_ingest=True` disables
    reuse and re-ingests every item (methodology default for published,
    multi-repetition batches per §5.3 — Day 2's budget-capped single-rep
    run uses the cheaper default)."""
    journal_path = Path(journal_path)
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    cache = ingestion_cache if ingestion_cache is not None else IngestionCache()
    provider_info = provider.info()

    usd_per_1k_tokens = (pricing or {}).get("usd_per_1k_tokens", 0.0)
    # Recall@k for k beyond top_k would mislabel a top-k-capped retrieval;
    # only compute what the retrieval depth can support.
    k_values = tuple(k for k in DEFAULT_K_VALUES if k <= top_k)
    successful_metrics: list[ItemMetric] = []
    total_cost_usd = 0.0
    infra_error_count = 0
    budget_truncated = False

    with journal_path.open("w") as journal:
        for rep in range(repetitions):
            for item in items:
                cache_key = ingestion_cache_key(provider_info, dataset_digest=dataset_digest, namespace=item.namespace)
                should_ingest = fresh_ingest or not cache.already_ingested(cache_key)

                record: dict = {
                    "run_id": run_id, "seed": seed, "rep": rep, "item_id": item.id, "ingested": should_ingest,
                }
                try:
                    add_latency_ms: float | None = None
                    settle_latency_ms: float | None = None
                    ingest_chars = 0
                    if should_ingest:
                        provider.reset(item.namespace)  # namespace hygiene, outside the timed window
                        add_start = time.perf_counter()
                        ingest_chars = _add_sessions(provider, item)
                        add_latency_ms = (time.perf_counter() - add_start) * 1000
                        # Accept-only providers finish ingestion here; timed
                        # apart so add/search percentiles stay pure (§8 Day 3
                        # latency-semantics contract in providers/base.py).
                        settle_start = time.perf_counter()
                        provider.settle(item.namespace)
                        settle_latency_ms = (time.perf_counter() - settle_start) * 1000
                        cache.mark_ingested(cache_key)

                    records, search_latency_ms = _search(provider, item, top_k=top_k)
                except ProviderError as exc:
                    infra_error_count += 1
                    record.update(status="infra_error", error=str(exc))
                    journal.write(json.dumps(record) + "\n")
                    journal.flush()  # a crash must not eat completed items (§5.3 resumability)
                    continue

                retrieved_contents = [r.content for r in records]
                # Verbatim evidence metrics only make sense against stores
                # that return source text; abstractive stores (distilled
                # memories) report N/A, never 0.0 (providers/base.py,
                # docs/METHODOLOGY_NOTES.md). Empty gold -> None everywhere.
                verbatim_gold = item.gold_evidence if provider.memory_representation == "extractive" else []
                metric = compute_item_metric(
                    item.id, retrieved_contents, verbatim_gold, add_latency_ms, search_latency_ms,
                    k_values=k_values,
                )
                total_chars = ingest_chars + len(item.question)
                cost_usd = estimate_cost_usd(total_chars, usd_per_1k_tokens=usd_per_1k_tokens)
                total_cost_usd += cost_usd
                successful_metrics.append(metric)
                record.update(
                    status="ok",
                    memory_representation=provider.memory_representation,
                    verbatim_recall_at_k=metric.verbatim_recall_at_k,
                    verbatim_ndcg_at_k=metric.verbatim_ndcg_at_k,
                    verbatim_reciprocal_rank=metric.verbatim_reciprocal_rank,
                    add_latency_ms=metric.add_latency_ms,
                    settle_latency_ms=settle_latency_ms,
                    search_latency_ms=metric.search_latency_ms,
                    cost_usd=cost_usd,
                )
                journal.write(json.dumps(record) + "\n")
                journal.flush()  # a crash must not eat completed items (§5.3 resumability)

                if budget_usd_max is not None and total_cost_usd > budget_usd_max:
                    budget_truncated = True
                    break
            if budget_truncated:
                break

    return RunResult(
        run_id=run_id,
        seed=seed,
        metrics=aggregate_run(successful_metrics, k_values=k_values),
        total_cost_usd=total_cost_usd,
        budget_truncated=budget_truncated,
        infra_error_count=infra_error_count,
        n_items_attempted=len(successful_metrics) + infra_error_count,
    )
