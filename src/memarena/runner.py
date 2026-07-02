from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

from memarena.datasets.base import QAItem
from memarena.errors import ProviderError
from memarena.metrics.deterministic import (
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


def _run_one_attempt(provider: MemoryProvider, item: QAItem, *, top_k: int) -> tuple[ItemMetric, list, int]:
    ingest_chars = 0
    add_start = time.perf_counter()
    for session in item.sessions:
        provider.add(item.namespace, session.messages, session_id=session.session_id, timestamp=session.timestamp)
        ingest_chars += sum(len(m["content"]) for m in session.messages)
    add_latency_ms = (time.perf_counter() - add_start) * 1000

    search_start = time.perf_counter()
    records = provider.search(item.namespace, item.question, top_k=top_k)
    search_latency_ms = (time.perf_counter() - search_start) * 1000

    retrieved_contents = [record.content for record in records]
    metric = compute_item_metric(
        item.id, retrieved_contents, item.gold_evidence, add_latency_ms, search_latency_ms,
    )
    total_chars = ingest_chars + len(item.question)
    return metric, records, total_chars


def run(
    provider: MemoryProvider,
    items: list[QAItem],
    *,
    run_id: str,
    seed: int,
    repetitions: int = 1,
    top_k: int = 5,
    budget_usd_max: float | None = None,
    pricing: dict | None = None,
    journal_path: str | Path,
) -> RunResult:
    """Runner v0 (§5.3, §8 Day 1): seeded item order via the caller-supplied
    `items` list, JSONL journal per attempt, budget guard that hard-stops
    with partial results, Level-1 deterministic metrics only."""
    journal_path = Path(journal_path)
    journal_path.parent.mkdir(parents=True, exist_ok=True)

    usd_per_1k_tokens = (pricing or {}).get("usd_per_1k_tokens", 0.0)
    successful_metrics: list[ItemMetric] = []
    total_cost_usd = 0.0
    infra_error_count = 0
    budget_truncated = False

    with journal_path.open("w") as journal:
        for rep in range(repetitions):
            for item in items:
                provider.reset(item.namespace)
                record: dict = {"run_id": run_id, "seed": seed, "rep": rep, "item_id": item.id}
                try:
                    metric, _, total_chars = _run_one_attempt(provider, item, top_k=top_k)
                except ProviderError as exc:
                    infra_error_count += 1
                    record.update(status="infra_error", error=str(exc))
                    journal.write(json.dumps(record) + "\n")
                    continue

                cost_usd = estimate_cost_usd(total_chars, usd_per_1k_tokens=usd_per_1k_tokens)
                total_cost_usd += cost_usd
                successful_metrics.append(metric)
                record.update(
                    status="ok",
                    recall_at_k=metric.recall_at_k,
                    reciprocal_rank=metric.reciprocal_rank,
                    add_latency_ms=metric.add_latency_ms,
                    search_latency_ms=metric.search_latency_ms,
                    cost_usd=cost_usd,
                )
                journal.write(json.dumps(record) + "\n")

                if budget_usd_max is not None and total_cost_usd > budget_usd_max:
                    budget_truncated = True
                    break
            if budget_truncated:
                break

    return RunResult(
        run_id=run_id,
        seed=seed,
        metrics=aggregate_run(successful_metrics),
        total_cost_usd=total_cost_usd,
        budget_truncated=budget_truncated,
        infra_error_count=infra_error_count,
        n_items_attempted=len(successful_metrics) + infra_error_count,
    )
