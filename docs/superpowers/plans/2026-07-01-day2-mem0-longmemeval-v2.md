# Day 2: Mem0 adapter + LongMemEval-V2 loader — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Execution note (this run):** executed inline in the same session that researched and wrote this plan (superpowers:executing-plans), because the research context (real dataset schema, real Mem0 client behavior) is expensive to reconstruct and does not transfer cheaply to a fresh subagent.

**Goal:** Ship §8 Day 2 of `../memarena_spec.md`: dataset/provider license audit, a real LongMemEval-V2 loader, a real Mem0 adapter, an ingestion cache, and a constant-reader answering-layer module — then run baseline_rag vs. mem0 on a 100-item LongMemEval-V2 stratified sample with deterministic (Level-1) metrics, cost metered, `budget_usd_max=10` respected.

**Architecture:** Extends the Day-1 skeleton (`runner.py`, `providers/base.py`, `datasets/base.py`, `metrics/deterministic.py`) with one new dataset loader, one new provider adapter, a small ingestion-cache module, and a standalone answering-layer module. No existing Day-1 public interface is broken; `runner.run()` gains two new optional parameters.

**Tech Stack:** Same as Day 1 (`httpx`, `tenacity`, `pydantic`, stdlib `sqlite3` not needed yet). New runtime dependency: `mem0ai==2.0.11` (pinned, confirmed on PyPI 2026-07-01). New dev-time-only network use (not part of `pytest`): HuggingFace Hub HTTP downloads.

## Global Constraints

- Python ≥3.11, existing `ruff` config (`select = ["E","F","I","UP","B"]`, line-length 120) — all new code must pass `ruff check`.
- `pytest` must stay 100% offline / free — no test may hit the network or spend money. All network-touching code (dataset download, Mem0 client, OpenAI client) must accept an injectable fake, mirroring `baseline_rag.py`'s `embed_fn` pattern.
- Never redistribute dataset content (spec §5.0.4). We only cache origin-downloaded artifacts locally (gitignored `results/`-like cache dir, not committed).
- `budget_usd_max=10` must be respected end-to-end for the real Day-2 run.
- Never fabricate ground truth. LongMemEval-V2's real schema (see Task 1 findings) has **no evidence-span field** — `gold_evidence` for every V2 item is `[]`, and Recall@k/MRR must render as **N/A**, not `0.0`, for this dataset. Do not invent evidence-matching heuristics.
- Mem0's real `/v3/memories/add/` endpoint is asynchronous (`{"event_id": ..., "status": "PENDING"}`), confirmed empirically 2026-07-01: a memory took ~5s to become visible via `get_all`. The adapter's `add()` must poll until the write is visible before returning, so it is a true synchronous façade (Appendix A rule) and so `search()` immediately after `add()` is fair.
- Constant reader model for the answering layer: pin `gpt-5-mini-2025-08-07` (dated snapshot, confirmed available on the account's OpenAI key 2026-07-01). Embedding model stays `text-embedding-3-small` (Day-1 baseline default, unchanged).

---

## Task 1: `datasets/LICENSES.md` + Mem0 tier documentation

**Files:**
- Create: `datasets/LICENSES.md`
- Modify: `.env.example` (add `MEM0_API_KEY=`)

**Research findings to record (already verified live, 2026-07-01 — do not re-derive, just write them up):**

1. **LongMemEval-V2**
   - Code repo: `github.com/xiaowu0162/LongMemEval-V2`, license Apache-2.0 (confirmed via `LICENSE` file).
   - Data: hosted on Hugging Face `xiaowu0162/longmemeval-v2`, dataset revision (commit sha) `f152293e235517d504809563c833d7190b8c713b`, license Apache-2.0 (confirmed via dataset card `cardData.license` and the dataset's own `LICENSE`/`DATA_CARD.md` files).
   - Real shape is **not** chat conversations: `questions.jsonl` (451 questions: `id, domain[web|enterprise], environment, question_type, question, image, answer, eval_function`) + `trajectories.jsonl` (1,870 web-agent trajectories, 1.2GB, LFS, sha256 `363cec9a8e87aa8d9101ce4e600aadbf7031d674056ebe4f969e8424abc5f3c6` per both the HF LFS oid and `checksums.sha256`) + two haystack tiers mapping `question_id → [trajectory_id, ...]`, shared per-domain (`haystacks/lme_v2_small.json`: 100 trajectories/domain; `lme_v2_medium.json`: up to 500/domain).
   - No evidence-span field exists anywhere in the schema — grading is via `eval_function` against a final boxed answer, not span-matching. `eval_function` distribution (451 questions): `norm_phrase_set_match` 200, `llm_abstention_checker` 128, `mc_choice_match` 68, `llm_gotchas_checker` 28, `norm_phrase_set_match_ordered` 26, `mc_choice_set_match` 1. The first four+last are deterministic string checkers; the two `llm_*` ones need a calibrated judge (Day 4 work).
   - `question_type` values and their abstention semantics: `static-environment` (134), `dynamic-environment` (86), `procedure` (74), `static-environment-abs` (55), `dynamic-environment-abs` (41), `procedure-abs` (32), `errors-gotchas` (29). Types ending in `-abs` map 1:1 to `llm_abstention_checker` — treat as `answerable=False`; all others `answerable=True`.
   - **Decision:** we run the *real* V2 dataset (preserves the spec's "fresh, no incumbent" thesis, §1) but Recall@k/MRR are not computable without evidence spans, so Day 2 reports them as N/A for this dataset while latency, cost, and ingestion throughput are fully real. Full accuracy grading via `eval_function` is scoped to Day 4 (judge work) since ~35% of items need an LLM checker anyway and it's cleaner to grade all of them under one calibrated mechanism.

2. **LoCoMo** (`github.com/snap-research/locomo`): license is **CC BY-NC 4.0** (Attribution-NonCommercial), confirmed via `LICENSE.txt`. Matches the spec's prediction (§5.6). Per spec's own risk mitigation (R1): running it and publishing aggregate *scores* is defensible, redistributing the data is not. **No loader built this Day 2** — out of scope (Day 2 build list is V2 + Mem0 only); this is a license-audit record for later.

3. **MemoryAgentBench** (`github.com/HUST-AI-HYZ/MemoryAgentBench`): license is **MIT**, confirmed via GitHub's license API. Fully permissive. No loader built this Day 2 either.

4. **Mem0 free tier** (confirmed via `mem0.ai/pricing` and the live account's `client.get_project()` call against the real `MEM0_API_KEY` in `.env`, project `default-project`, owner `tirelli@gmail.com`): **Hobby/free tier = 10,000 memory-add requests/month, 1,000 retrieval calls/month, 1 project.** Day 2's planned real run uses at most ~24 `add()` calls and ~200 `search()` calls (2 domains × 6 trajectories × 2 providers for adds; 100 items × 2 providers for searches) — comfortably inside quota with >99% headroom. Record this arithmetic in the doc.

- [ ] **Step 1: Write `datasets/LICENSES.md`**

```markdown
# Dataset & Provider License Audit (§5.6)

Verified 2026-07-01. Record updated whenever a new dataset/provider is added.

## LongMemEval-V2

- Code: https://github.com/xiaowu0162/LongMemEval-V2 — **Apache-2.0**.
- Data: https://huggingface.co/datasets/xiaowu0162/longmemeval-v2 — **Apache-2.0**,
  pinned revision `f152293e235517d504809563c833d7190b8c713b`.
- Redistribution: none. The loader downloads from origin at the pinned revision and
  verifies sha256 against the dataset's own `checksums.sha256`; this repo ships zero
  dataset bytes.
- Schema is **not** chat conversation data — it is 451 questions graded against
  1,870 web-agent trajectories (WebArena/ServiceNow-style), with two shared
  per-domain "haystacks" of candidate trajectories. There is no evidence-span
  field. See `src/memarena/datasets/longmemeval_v2.py` module docstring for the
  full field-level mapping this harness uses.
- Consequence for metrics: Recall@k / MRR (Level-1 retrieval metrics, §5.7) are
  **not computable** for this dataset without evidence spans that do not exist
  in the source data — we report them as N/A rather than fabricate an
  evidence-matching heuristic. Latency, cost, and ingestion throughput are real
  and fully reported. Answer-correctness grading (`eval_function`-based, partly
  deterministic / partly LLM-judge-based) is scoped to Day 4.

## LoCoMo

- https://github.com/snap-research/locomo — **CC BY-NC 4.0** (non-commercial).
  Confirmed via `LICENSE.txt`, matches the spec's prediction (§5.6).
- Per spec R1: running evaluations and publishing aggregate *scores* is
  defensible under this license; redistributing the underlying conversations
  is not, and we never do that for any dataset regardless of license.
- **No loader implemented as of Day 2.** Deferred; not on the Day-2 build list.

## MemoryAgentBench

- https://github.com/HUST-AI-HYZ/MemoryAgentBench — **MIT**. Fully permissive.
- **No loader implemented as of Day 2.** Deferred; not on the Day-2 build list.

## Mem0 (provider, not a dataset)

- Free ("Hobby") tier, confirmed via https://mem0.ai/pricing and the live
  account (`client.get_project()`, project `default-project`,
  2026-07-01): **10,000 memory-add requests/month, 1,000 retrieval calls/month,
  1 project.**
- Day 2's real run against LongMemEval-V2 (100-item stratified sample, 2
  providers) issues at most ~24 `add()` calls (2 domains × up to 6 trajectories
  × 2 providers, since ingestion is cached per-namespace — see
  `src/memarena/cache.py`) and ~200 `search()` calls (100 items × 2 providers).
  Both are comfortably under quota (<1% and ~20% respectively, worst case).
  Tier used: **free/Hobby**, `mem0ai==2.0.11`.
```

- [ ] **Step 2: Add `MEM0_API_KEY` to `.env.example`**

```
OPENAI_API_KEY=
MEM0_API_KEY=
```

- [ ] **Step 3: Commit**

```bash
git add datasets/LICENSES.md .env.example
git commit -m "docs: dataset/provider license audit for Day 2 (§5.6)"
```

---

## Task 2: `metrics/deterministic.py` — N/A-safe aggregation + optional add-latency

**Files:**
- Modify: `src/memarena/metrics/deterministic.py`
- Test: `tests/test_metrics_deterministic.py` (extend existing file — read it first to match its style before editing)

**Why:** two Day-1 gaps block honest V2 reporting:
1. `mean_of_defined([])` returns `0.0`. When a dataset has zero items with gold evidence (V2, always), `RunMetrics.recall_at_k`/`mrr` would silently print `0.000` — indistinguishable from "the system got everything wrong." Must be `None` (N/A) instead.
2. `add_latency_ms` is a required `float` on `ItemMetric`. With the Task 5 ingestion cache, most V2 items reuse a namespace's ingestion and pay no add cost this item — that item's add latency must be excluded from the p50/p95, not counted as `0.0` (which would skew percentiles down and misrepresent real ingestion cost).

**Interfaces:**
- Consumes: nothing new.
- Produces: `ItemMetric.add_latency_ms: float | None` (was `float`); `RunMetrics.recall_at_k: dict[int, float | None]`, `RunMetrics.mrr: float | None`, `RunMetrics.add_latency_p50_ms: float | None`, `RunMetrics.add_latency_p95_ms: float | None`. `compute_item_metric(..., add_latency_ms: float | None, ...)` — the `None` case means "this item reused a prior namespace's ingestion, no add happened this item."

- [ ] **Step 1: Write the failing tests**

```python
class TestMeanOfDefinedAllNone:
    def test_all_none_returns_none_not_zero(self):
        assert mean_of_defined([None, None]) is None

    def test_empty_list_returns_none(self):
        assert mean_of_defined([]) is None

    def test_mixed_none_and_values_ignores_none(self):
        assert mean_of_defined([None, 1.0, 0.0]) == 0.5


class TestPercentileOfDefined:
    def test_ignores_none_values(self):
        assert percentile_of_defined([None, 10.0, 20.0, None], 50) == 15.0

    def test_all_none_returns_none(self):
        assert percentile_of_defined([None, None], 50) is None


class TestAggregateRunWithNoGoldEvidence:
    def test_recall_and_mrr_are_none_when_no_item_has_gold_evidence(self):
        items = [
            compute_item_metric("i1", ["x"], [], add_latency_ms=5.0, search_latency_ms=10.0),
            compute_item_metric("i2", ["y"], [], add_latency_ms=None, search_latency_ms=12.0),
        ]
        metrics = aggregate_run(items)
        assert metrics.recall_at_k[5] is None
        assert metrics.mrr is None
        assert metrics.n_scored_items == 0

    def test_add_latency_percentiles_ignore_reused_ingestion_items(self):
        items = [
            compute_item_metric("i1", ["x"], ["x"], add_latency_ms=100.0, search_latency_ms=1.0),
            compute_item_metric("i2", ["x"], ["x"], add_latency_ms=None, search_latency_ms=1.0),
            compute_item_metric("i3", ["x"], ["x"], add_latency_ms=None, search_latency_ms=1.0),
        ]
        metrics = aggregate_run(items)
        assert metrics.add_latency_p50_ms == 100.0  # only the one real ingest counts
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_metrics_deterministic.py -v`
Expected: FAIL — `percentile_of_defined` not defined; `mean_of_defined([])` currently returns `0.0` not `None`; `compute_item_metric` doesn't accept `add_latency_ms=None`.

- [ ] **Step 3: Implement**

Replace `mean_of_defined` and add `percentile_of_defined`:

```python
def mean_of_defined(values: list[float | None]) -> float | None:
    defined = [v for v in values if v is not None]
    if not defined:
        return None
    return sum(defined) / len(defined)


def percentile_of_defined(values: list[float | None], p: float) -> float | None:
    defined = [v for v in values if v is not None]
    if not defined:
        return None
    return percentile(defined, p)
```

Update `ItemMetric`:

```python
@dataclass(frozen=True)
class ItemMetric:
    item_id: str
    recall_at_k: dict[int, float | None]
    reciprocal_rank: float | None
    add_latency_ms: float | None
    search_latency_ms: float
```

Update `compute_item_metric` signature to accept `add_latency_ms: float | None` (no other change to its body — it just stores what it's given).

Update `RunMetrics` and `aggregate_run`:

```python
@dataclass(frozen=True)
class RunMetrics:
    recall_at_k: dict[int, float | None]
    mrr: float | None
    add_latency_p50_ms: float | None
    add_latency_p95_ms: float | None
    search_latency_p50_ms: float
    search_latency_p95_ms: float
    n_items: int
    n_scored_items: int


def aggregate_run(items: list[ItemMetric], *, k_values: tuple[int, ...] = (1, 3, 5, 10)) -> RunMetrics:
    add_latencies = [item.add_latency_ms for item in items]
    search_latencies = [item.search_latency_ms for item in items]
    return RunMetrics(
        recall_at_k={k: mean_of_defined([item.recall_at_k[k] for item in items]) for k in k_values},
        mrr=mean_of_defined([item.reciprocal_rank for item in items]),
        add_latency_p50_ms=percentile_of_defined(add_latencies, 50),
        add_latency_p95_ms=percentile_of_defined(add_latencies, 95),
        search_latency_p50_ms=percentile(search_latencies, 50),
        search_latency_p95_ms=percentile(search_latencies, 95),
        n_items=len(items),
        n_scored_items=sum(1 for item in items if item.reciprocal_rank is not None),
    )
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/test_metrics_deterministic.py -v`
Expected: PASS, all tests including pre-existing ones.

- [ ] **Step 5: Fix the now-broken call sites** (`runner.py`, `cli.py`) — see Tasks 4 and 6 below; do not commit this task alone until Task 4 lands, since `runner.py` currently passes a non-optional float and `cli.py`'s `_print_result` does `f"{metrics.mrr:.3f}"` which crashes on `None`. Run `pytest` (full suite) here to confirm the break is visible:

Run: `pytest -q`
Expected: FAIL in `test_runner.py` / `test_cli.py` (or wherever the format string chokes) — confirms the ripple is real, will be fixed in Task 4.

- [ ] **Step 6: Commit** (bundled with Task 4's commit, since the two are not independently green — see Task 4 Step 6)

---

## Task 3: `src/memarena/cache.py` — in-run ingestion cache

**Files:**
- Create: `src/memarena/cache.py`
- Test: `tests/test_cache.py`

**Why:** LongMemEval-V2's two haystacks are shared per-domain across dozens of questions (SCHEMA.md: "within each domain, all questions share one 100-trajectory haystack"). Re-ingesting the same trajectories for every question would mean ~1,200 Mem0 `add()` calls instead of ~24, each with a multi-second async-settle poll (Task 6) — turning a ~5 minute run into ~2+ hours for no benefit. This module tracks, within a single `runner.run()` invocation, which `(provider, config, dataset, namespace)` combinations have already been ingested.

**Scope decision (documented, not a placeholder):** this is an **in-process, non-persistent** cache — it does not survive across separate CLI invocations. A persistent sqlite-backed cache (spec §5.2's `cache.py: sqlite content-hash caches: ingestion + judge`) is deferred; persisting ingestion state across runs is only safe for providers with real external storage (Mem0) and actively unsafe for in-memory ones (`baseline_rag`, which starts every process with an empty store) unless the loader also checks provider-side idempotency — not needed yet at this scale.

**Interfaces:**
- Consumes: nothing.
- Produces: `IngestionCache` class with `key(provider_info: ProviderInfo, dataset_digest: str, namespace: str) -> str` (module-level function) and instance methods `already_ingested(key: str) -> bool`, `mark_ingested(key: str) -> None`. Used by `runner.py` (Task 4).

- [ ] **Step 1: Write the failing test**

```python
import pytest

from memarena.cache import IngestionCache, ingestion_cache_key
from memarena.providers.base import ProviderInfo


def _info(name="p", client_version="1.0", config_digest="abc"):
    return ProviderInfo(name=name, client_version=client_version, config_digest=config_digest,
                         pricing_model="per_token")


class TestIngestionCacheKey:
    def test_key_differs_by_namespace(self):
        k1 = ingestion_cache_key(_info(), dataset_digest="d1", namespace="ns1")
        k2 = ingestion_cache_key(_info(), dataset_digest="d1", namespace="ns2")
        assert k1 != k2

    def test_key_differs_by_config_digest(self):
        k1 = ingestion_cache_key(_info(config_digest="abc"), dataset_digest="d1", namespace="ns1")
        k2 = ingestion_cache_key(_info(config_digest="xyz"), dataset_digest="d1", namespace="ns1")
        assert k1 != k2

    def test_key_differs_by_dataset_digest(self):
        k1 = ingestion_cache_key(_info(), dataset_digest="d1", namespace="ns1")
        k2 = ingestion_cache_key(_info(), dataset_digest="d2", namespace="ns1")
        assert k1 != k2

    def test_key_is_stable(self):
        k1 = ingestion_cache_key(_info(), dataset_digest="d1", namespace="ns1")
        k2 = ingestion_cache_key(_info(), dataset_digest="d1", namespace="ns1")
        assert k1 == k2


class TestIngestionCache:
    def test_unseen_key_is_not_ingested(self):
        cache = IngestionCache()
        assert cache.already_ingested("k1") is False

    def test_marked_key_is_ingested(self):
        cache = IngestionCache()
        cache.mark_ingested("k1")
        assert cache.already_ingested("k1") is True

    def test_marking_is_isolated_per_instance(self):
        cache_a, cache_b = IngestionCache(), IngestionCache()
        cache_a.mark_ingested("k1")
        assert cache_b.already_ingested("k1") is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cache.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'memarena.cache'`

- [ ] **Step 3: Implement**

```python
# src/memarena/cache.py
from __future__ import annotations

from memarena.providers.base import ProviderInfo


def ingestion_cache_key(info: ProviderInfo, *, dataset_digest: str, namespace: str) -> str:
    """Cache key for 'has this namespace already been ingested this run'
    (§5.3). Changing provider config, client version, or dataset revision
    must invalidate the cache — all four are part of the key."""
    return f"{info.name}:{info.client_version}:{info.config_digest}:{dataset_digest}:{namespace}"


class IngestionCache:
    """In-run (non-persistent) ingestion cache — see module scope note in
    the Day 2 plan. Tracks which (provider, config, dataset, namespace)
    combinations have already had `reset()`+`add()` performed this run."""

    def __init__(self) -> None:
        self._seen: set[str] = set()

    def already_ingested(self, key: str) -> bool:
        return key in self._seen

    def mark_ingested(self, key: str) -> None:
        self._seen.add(key)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_cache.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/memarena/cache.py tests/test_cache.py
git commit -m "feat: add in-run ingestion cache (§5.3)"
```

---

## Task 4: `runner.py` — wire in ingestion cache + dataset digest + optional add latency

**Files:**
- Modify: `src/memarena/runner.py`
- Modify: `src/memarena/cli.py` (thread `dataset_digest` through)
- Test: `tests/test_runner.py` (extend)

**Interfaces:**
- Consumes: `IngestionCache`, `ingestion_cache_key` from Task 3; `ItemMetric`/`RunMetrics` with optional fields from Task 2.
- Produces: `run(provider, items, *, run_id, seed, repetitions=1, top_k=5, budget_usd_max=None, pricing=None, journal_path, dataset_digest, ingestion_cache=None, fresh_ingest=False) -> RunResult`. New: `dataset_digest: str` (required, no default — every caller must supply it, matching the reproducibility principle §5.0.2), `ingestion_cache: IngestionCache | None = None` (a fresh one is created if not passed), `fresh_ingest: bool = False`.

- [ ] **Step 1: Write the failing tests** (add to `tests/test_runner.py`)

```python
from memarena.cache import IngestionCache, ingestion_cache_key


class TestIngestionReuse:
    def test_second_item_sharing_namespace_reuses_ingestion(self, tmp_path):
        provider = FakeProvider()
        items = [
            _item("i1", "shared-ns", "fact A", "fact", ["fact A"]),
            _item("i2", "shared-ns", "fact A", "fact", ["fact A"]),
        ]
        journal_path = tmp_path / "journal.jsonl"

        result = run(
            provider, items, run_id="t", seed=42, repetitions=1, top_k=5,
            budget_usd_max=None, pricing=None, journal_path=journal_path,
            dataset_digest="dset1",
        )

        # only one add() worth of content in the store — reset+add ran once
        assert provider.store["shared-ns"] == ["fact A"]
        lines = [json.loads(line) for line in journal_path.read_text().strip().splitlines()]
        assert lines[0]["ingested"] is True
        assert lines[1]["ingested"] is False
        assert lines[1]["add_latency_ms"] is None
        assert result.metrics.n_items == 2  # both items still scored via search

    def test_fresh_ingest_forces_reingestion_every_item(self, tmp_path):
        provider = FakeProvider()
        items = [
            _item("i1", "shared-ns", "fact A", "fact", ["fact A"]),
            _item("i2", "shared-ns", "fact A", "fact", ["fact A"]),
        ]
        journal_path = tmp_path / "journal.jsonl"

        run(
            provider, items, run_id="t", seed=42, repetitions=1, top_k=5,
            budget_usd_max=None, pricing=None, journal_path=journal_path,
            dataset_digest="dset1", fresh_ingest=True,
        )
        lines = [json.loads(line) for line in journal_path.read_text().strip().splitlines()]
        assert lines[0]["ingested"] is True
        assert lines[1]["ingested"] is True

    def test_different_dataset_digest_does_not_reuse_across_separate_runs(self, tmp_path):
        provider = FakeProvider()
        items = [_item("i1", "shared-ns", "fact A", "fact", ["fact A"])]
        cache = IngestionCache()
        cache.mark_ingested(ingestion_cache_key(provider.info(), dataset_digest="other-dset", namespace="shared-ns"))

        result = run(
            provider, items, run_id="t", seed=42, repetitions=1, top_k=5,
            budget_usd_max=None, pricing=None, journal_path=tmp_path / "j.jsonl",
            dataset_digest="dset1", ingestion_cache=cache,
        )
        assert result.metrics.n_items == 1
        assert provider.store["shared-ns"] == ["fact A"]  # still ingested — different digest, cache miss
```

Also update every pre-existing `run(...)` call in `test_runner.py` to pass `dataset_digest="test-dataset"` (they'll break otherwise since it's a new required parameter).

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_runner.py -v`
Expected: FAIL — `run()` doesn't accept `dataset_digest`/`ingestion_cache`/`fresh_ingest` yet.

- [ ] **Step 3: Implement** — replace `runner.py` body:

```python
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

from memarena.cache import IngestionCache, ingestion_cache_key
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


def _ingest(provider: MemoryProvider, item: QAItem) -> int:
    """Reset the namespace and add every session. Returns total ingested chars."""
    provider.reset(item.namespace)
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
    successful_metrics: list[ItemMetric] = []
    total_cost_usd = 0.0
    infra_error_count = 0
    budget_truncated = False

    with journal_path.open("w") as journal:
        for rep in range(repetitions):
            for item in items:
                cache_key = ingestion_cache_key(provider_info, dataset_digest=dataset_digest, namespace=item.namespace)
                should_ingest = fresh_ingest or not cache.already_ingested(cache_key)

                record: dict = {"run_id": run_id, "seed": seed, "rep": rep, "item_id": item.id, "ingested": should_ingest}
                try:
                    add_latency_ms: float | None = None
                    ingest_chars = 0
                    if should_ingest:
                        add_start = time.perf_counter()
                        ingest_chars = _ingest(provider, item)
                        add_latency_ms = (time.perf_counter() - add_start) * 1000
                        cache.mark_ingested(cache_key)

                    records, search_latency_ms = _search(provider, item, top_k=top_k)
                except ProviderError as exc:
                    infra_error_count += 1
                    record.update(status="infra_error", error=str(exc))
                    journal.write(json.dumps(record) + "\n")
                    continue

                retrieved_contents = [r.content for r in records]
                metric = compute_item_metric(
                    item.id, retrieved_contents, item.gold_evidence, add_latency_ms, search_latency_ms,
                )
                total_chars = ingest_chars + len(item.question)
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
```

Update `cli.py`:
- In `run_command`, after `items = dataset_cls().load(...)`, add: `dataset_digest = dataset_cls().sha256()`.
- Pass `dataset_digest=dataset_digest` into the `run_experiment(...)` call.
- Fix `_print_result` to handle `None` metrics (N/A rendering):

```python
def _fmt(value: float | None, spec: str = ".3f") -> str:
    return "N/A" if value is None else format(value, spec)


def _print_result(provider_name: str, dataset_name: str, result: RunResult) -> None:
    metrics = result.metrics
    typer.echo(f"=== {provider_name} on {dataset_name} ===")
    typer.echo(f"Recall@5: {_fmt(metrics.recall_at_k.get(5))}")
    typer.echo(f"MRR: {_fmt(metrics.mrr)}")
    typer.echo(f"Add latency p50/p95 (ms): {_fmt(metrics.add_latency_p50_ms, '.1f')} / {_fmt(metrics.add_latency_p95_ms, '.1f')}")
    typer.echo(
        f"Search latency p50/p95 (ms): {metrics.search_latency_p50_ms:.1f} / {metrics.search_latency_p95_ms:.1f}"
    )
    budget_note = "truncated" if result.budget_truncated else "not truncated"
    typer.echo(f"Cost: ${result.total_cost_usd:.4f} (budget: {budget_note})")
    typer.echo(f"Items: {metrics.n_scored_items} scored ({result.infra_error_count} infra errors)")
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/test_runner.py tests/test_metrics_deterministic.py tests/test_cli.py -v`
Expected: PASS — including Task 2's tests, now unblocked.

- [ ] **Step 5: Run the full existing suite to catch ripple effects**

Run: `pytest -q`
Expected: PASS (fixes the Task 2 Step 5 break). `test_cli.py` may need a `dataset_digest`/smoke-set update — check `SmokeDatasetLoader.sha256()` is called correctly through the CLI path (it already exists from Day 1, no change needed there beyond the new call site).

- [ ] **Step 6: Commit**

```bash
git add src/memarena/runner.py src/memarena/cli.py src/memarena/metrics/deterministic.py tests/test_runner.py tests/test_metrics_deterministic.py tests/test_cli.py
git commit -m "feat: ingestion cache + N/A-safe metrics in the runner (§5.3, §5.7)"
```

---

## Task 5: LongMemEval-V2 dataset loader

**Files:**
- Create: `src/memarena/datasets/longmemeval_v2.py`
- Create: `tests/fixtures/lme_v2_questions_sample.jsonl` (5-10 line fixture, hand-copied from real downloaded data — see Step 0)
- Create: `tests/fixtures/lme_v2_trajectories_sample.jsonl` (matching trajectory fixture, trimmed to 2-3 short states each)
- Create: `tests/fixtures/lme_v2_haystack_small_sample.json`
- Test: `tests/test_longmemeval_v2.py`

**Real schema (confirmed live 2026-07-01, do not re-derive):**

`questions.jsonl` line: `{"id": str, "domain": "web"|"enterprise", "environment": str, "question_type": str, "question": str, "image": str|null, "answer": str, "eval_function": str}`.

`trajectories.jsonl` line: `{"id": str, "domain": str, "environment": str, "goal": str, "outcome": "success"|"failure", "start_url": str, "states": [{"state_index": int, "step": int|null, "url": str, "action": str|null, "thought": str|null, "accessibility_tree": str, "screenshot": str}]}`. No timestamp field anywhere.

`haystacks/lme_v2_small.json`: `{question_id: [trajectory_id, ...]}` — within a domain, every question's array is the *same* 100 ids (confirmed in SCHEMA.md and by inspection).

**Design decisions (documented caps, not silent truncation):**
- Download `questions.jsonl` (286KB) and `haystacks/lme_v2_small.json` (822KB) in full; verify each against `checksums.sha256` published at the same revision.
- Do **not** download `haystacks/lme_v2_medium.json` or the full `trajectories.jsonl` eagerly. Stream `trajectories.jsonl` once (1.2GB, ~4-5 min at the ~4MB/s measured 2026-07-01), computing a running sha256 over the *entire* stream (verified against the published checksum) while writing only the lines whose `id` is in the needed set to a local cache file — everything else is discarded, never persisted.
- "Needed set" = the first `TRAJECTORIES_PER_DOMAIN = 6` ids (preserving haystack order) from each domain's shared small-haystack array — i.e., at most 12 trajectories total regardless of sample size, because the haystack is domain-shared. This bounds ingestion cost/time; a larger run (full 100- or 500-trajectory haystack) is future paid-tier work, documented in the module docstring.
- Per trajectory, cap `STATES_PER_TRAJECTORY = 15` states and `ACCESSIBILITY_TREE_CHARS = 500` chars/state — real content, truncated, not fabricated.
- `gold_evidence` is always `[]` (no evidence spans in this dataset — see Task 1 finding). `answerable = not question_type.endswith("-abs")`.
- `namespace = f"lme_v2_{domain}"` — **two** namespaces total (`lme_v2_web`, `lme_v2_enterprise`), shared across every item of that domain, so the runner's ingestion cache (Task 3/4) ingests each domain's trajectories exactly once regardless of how many sampled questions belong to it.
- Each trajectory becomes one `Session`: `session_id=trajectory["id"]`, `timestamp` = a synthetic sequential ISO8601 timestamp (`2026-01-01T00:00:00Z` + trajectory's index-in-haystack days) since the source has no real capture time — documented in the docstring as synthetic, not authoritative. `messages`: one `role="user"` message with `f"Goal: {goal}"`, then one `role="assistant"` message per (capped) state with `f"URL: {url}\nAction: {action}\nObservation: {accessibility_tree[:500]}"`.

**Interfaces:**
- Consumes: `DatasetLoader`, `QAItem`, `Session` from `datasets/base.py` (Task-independent, already exists).
- Produces: `LongMemEvalV2Loader(DatasetLoader)` with `name="longmemeval_v2"`, `origin_url="https://huggingface.co/datasets/xiaowu0162/longmemeval-v2"`, `revision="f152293e235517d504809563c833d7190b8c713b"`, `license="Apache-2.0"`, `redistributable=False`. Constructor takes `*, fetcher: Fetcher | None = None, cache_dir: Path | None = None` where `Fetcher` is a small injectable protocol (see Step 3) so tests never touch the network.

- [ ] **Step 0: Create the test fixtures from the real downloaded files** (one-time, by hand/script — not part of the loader code)

```bash
python3 - <<'PYEOF'
import json
lines = []
with open('/tmp/questions_full.jsonl') as f:
    for line in f:
        obj = json.loads(line)
        lines.append(obj)

# take 2 web + 2 enterprise, spanning an answerable and an -abs type each
picked = []
seen_types = set()
for obj in lines:
    key = (obj["domain"], obj["question_type"].endswith("-abs"))
    if key not in seen_types:
        picked.append(obj)
        seen_types.add(key)
    if len(picked) >= 6:
        break

with open('tests/fixtures/lme_v2_questions_sample.jsonl', 'w') as out:
    for obj in picked:
        out.write(json.dumps(obj) + "\n")

print([p["id"] for p in picked], [p["domain"] for p in picked])
PYEOF
```

Then hand-author `tests/fixtures/lme_v2_trajectories_sample.jsonl` with 2-3 short, self-contained trajectory objects (2-3 states each, short accessibility_tree strings) for `web` and `enterprise` domains using the **real field names** from SCHEMA.md (`id, domain, environment, goal, outcome, start_url, states[].{state_index,step,url,action,thought,accessibility_tree,screenshot}`) — content can be synthetic/short since this is a test fixture, not redistributed real data at scale. Author matching `tests/fixtures/lme_v2_haystack_small_sample.json` mapping each picked question's domain to the fixture trajectory ids for that domain (e.g. `{"web": ["traj-w1", "traj-w2"], "enterprise": ["traj-e1", "traj-e2"]}`, matching the loader's expected in-memory shape after parsing — see Step 3, the loader normalizes the real per-question-id haystack into a per-domain list internally since values are identical within a domain).

- [ ] **Step 1: Write the failing tests**

```python
import json

import pytest

from memarena.datasets.base import QAItem
from memarena.datasets.longmemeval_v2 import LongMemEvalV2Loader

FIXTURES = "tests/fixtures"


class FakeFetcher:
    """Test double standing in for the real HF-download fetcher — returns
    fixture bytes/paths instead of hitting the network."""

    def __init__(self, questions_path, trajectories_path, haystack_path):
        self.questions_path = questions_path
        self.trajectories_path = trajectories_path
        self.haystack_path = haystack_path

    def fetch_questions(self) -> str:
        return self.questions_path

    def fetch_haystack_small(self) -> str:
        return self.haystack_path

    def fetch_trajectories(self, needed_ids: set[str]) -> str:
        return self.trajectories_path


@pytest.fixture
def loader(tmp_path):
    fetcher = FakeFetcher(
        f"{FIXTURES}/lme_v2_questions_sample.jsonl",
        f"{FIXTURES}/lme_v2_trajectories_sample.jsonl",
        f"{FIXTURES}/lme_v2_haystack_small_sample.json",
    )
    return LongMemEvalV2Loader(fetcher=fetcher, cache_dir=tmp_path)


class TestLongMemEvalV2Loader:
    def test_declares_metadata(self, loader):
        assert loader.name == "longmemeval_v2"
        assert loader.license == "Apache-2.0"
        assert loader.redistributable is False
        assert loader.revision == "f152293e235517d504809563c833d7190b8c713b"

    def test_load_returns_qaitems(self, loader):
        items = loader.load()
        assert all(isinstance(i, QAItem) for i in items)
        assert len(items) > 0

    def test_gold_evidence_is_always_empty(self, loader):
        for item in loader.load():
            assert item.gold_evidence == []

    def test_abs_question_types_are_not_answerable(self, loader):
        for item in loader.load():
            if item.question_type.endswith("-abs"):
                assert item.answerable is False
            else:
                assert item.answerable is True

    def test_namespace_is_shared_per_domain(self, loader):
        items = loader.load()
        namespaces = {item.namespace for item in items}
        assert namespaces <= {"lme_v2_web", "lme_v2_enterprise"}

    def test_sessions_carry_capped_trajectory_content(self, loader):
        items = loader.load()
        item = items[0]
        assert len(item.sessions) > 0
        for session in item.sessions:
            assert session.messages[0]["role"] == "user"
            assert session.messages[0]["content"].startswith("Goal:")

    def test_sample_is_deterministic_given_seed(self, loader):
        a = loader.load(sample=2, seed=7)
        b = loader.load(sample=2, seed=7)
        assert [i.id for i in a] == [i.id for i in b]

    def test_sha256_is_stable_hex_digest(self, loader):
        digest_a = loader.sha256()
        digest_b = loader.sha256()
        assert digest_a == digest_b
        assert len(digest_a) == 64
        int(digest_a, 16)
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_longmemeval_v2.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement the loader**

```python
# src/memarena/datasets/longmemeval_v2.py
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Protocol

from memarena.datasets.base import DatasetLoader, QAItem, Session

# --- Real schema, confirmed 2026-07-01 against the live origin (do not
# re-derive without re-checking — see datasets/LICENSES.md):
#   questions.jsonl: {id, domain[web|enterprise], environment, question_type,
#                     question, image, answer, eval_function}
#   trajectories.jsonl: {id, domain, environment, goal, outcome, start_url,
#                        states:[{state_index, step, url, action, thought,
#                        accessibility_tree, screenshot}]}
#   haystacks/lme_v2_small.json: {question_id: [trajectory_id, ...]} — all
#   questions in one domain share the same 100-id array.
#
# There is no evidence-span field anywhere in this dataset — QAItem.gold_evidence
# is always []; Recall@k/MRR report as N/A for this dataset (see
# metrics/deterministic.py's None-safe aggregation). Grading is via
# `eval_function` against a final boxed answer, scoped to Day 4 (judge work).
#
# Scope caps (documented, not silent truncation — a larger paid run can raise
# these): only the first TRAJECTORIES_PER_DOMAIN ids per domain's shared
# haystack are ingested; each trajectory is capped to STATES_PER_TRAJECTORY
# states and ACCESSIBILITY_TREE_CHARS chars/state.
TRAJECTORIES_PER_DOMAIN = 6
STATES_PER_TRAJECTORY = 15
ACCESSIBILITY_TREE_CHARS = 500

REVISION = "f152293e235517d504809563c833d7190b8c713b"
ORIGIN_URL = "https://huggingface.co/datasets/xiaowu0162/longmemeval-v2"


class Fetcher(Protocol):
    """Injectable network boundary — the real implementation downloads from
    `ORIGIN_URL` at `REVISION` with sha256 verification against the origin's
    `checksums.sha256`; tests supply a fixture-backed fake (§ Day 2 plan)."""

    def fetch_questions(self) -> str: ...
    def fetch_haystack_small(self) -> str: ...
    def fetch_trajectories(self, needed_ids: set[str]) -> str: ...


def _question_type_answerable(question_type: str) -> bool:
    return not question_type.endswith("-abs")


def _synthetic_timestamp(index: int) -> str:
    """The source dataset has no capture timestamps. This assigns a
    synthetic, deterministic sequence — documented as non-authoritative."""
    day = 1 + (index % 27)
    return f"2026-01-{day:02d}T00:00:00Z"


def _trajectory_to_session(trajectory: dict, *, index: int) -> Session:
    lines = [{"role": "user", "content": f"Goal: {trajectory['goal']}"}]
    for state in trajectory["states"][:STATES_PER_TRAJECTORY]:
        tree = (state.get("accessibility_tree") or "")[:ACCESSIBILITY_TREE_CHARS]
        content = f"URL: {state['url']}\nAction: {state.get('action')}\nObservation: {tree}"
        lines.append({"role": "assistant", "content": content})
    return Session(session_id=trajectory["id"], timestamp=_synthetic_timestamp(index), messages=lines)


class LongMemEvalV2Loader(DatasetLoader):
    """Real LongMemEval-V2 (§8 Day 2). See module docstring above and
    datasets/LICENSES.md for the full license/schema/scope-cap record."""

    name = "longmemeval_v2"
    origin_url = ORIGIN_URL
    revision = REVISION
    license = "Apache-2.0"
    redistributable = False

    def __init__(self, *, fetcher: Fetcher | None = None, cache_dir: Path | str | None = None):
        self._fetcher = fetcher or _default_fetcher(cache_dir)
        self._questions: list[dict] | None = None
        self._sessions_by_domain: dict[str, list[Session]] | None = None

    def _load_raw(self) -> None:
        if self._questions is not None:
            return
        questions_path = self._fetcher.fetch_questions()
        with open(questions_path) as f:
            self._questions = [json.loads(line) for line in f if line.strip()]

        haystack_path = self._fetcher.fetch_haystack_small()
        with open(haystack_path) as f:
            haystack = json.load(f)

        domain_by_question = {q["id"]: q["domain"] for q in self._questions}
        needed_by_domain: dict[str, list[str]] = {}
        for question_id, trajectory_ids in haystack.items():
            domain = domain_by_question.get(question_id)
            if domain and domain not in needed_by_domain:
                needed_by_domain[domain] = trajectory_ids[:TRAJECTORIES_PER_DOMAIN]

        needed_ids = {tid for ids in needed_by_domain.values() for tid in ids}
        trajectories_path = self._fetcher.fetch_trajectories(needed_ids)
        trajectories_by_id: dict[str, dict] = {}
        with open(trajectories_path) as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                if obj["id"] in needed_ids:
                    trajectories_by_id[obj["id"]] = obj

        self._sessions_by_domain = {}
        for domain, ids in needed_by_domain.items():
            sessions = [
                _trajectory_to_session(trajectories_by_id[tid], index=i)
                for i, tid in enumerate(ids) if tid in trajectories_by_id
            ]
            self._sessions_by_domain[domain] = sessions

    def sha256(self) -> str:
        self._load_raw()
        canonical = json.dumps(self._questions, sort_keys=True).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

    def load(self, *, sample: int | None = None, seed: int = 42,
             stratify_by: str | None = "question_type") -> list[QAItem]:
        self._load_raw()
        items = [self._to_qaitem(q) for q in self._questions if q["domain"] in self._sessions_by_domain]
        if sample is None or sample >= len(items):
            return sorted(items, key=lambda i: i.id)
        return self._stratified_sample(items, sample, seed, stratify_by)

    def _to_qaitem(self, q: dict) -> QAItem:
        return QAItem(
            id=q["id"],
            namespace=f"lme_v2_{q['domain']}",
            sessions=self._sessions_by_domain[q["domain"]],
            question=q["question"],
            gold_evidence=[],
            question_type=q["question_type"],
            answerable=_question_type_answerable(q["question_type"]),
            gold_answer=q["answer"],
        )

    @staticmethod
    def _stratified_sample(items: list[QAItem], sample: int, seed: int,
                            stratify_by: str | None) -> list[QAItem]:
        import random
        rng = random.Random(seed)
        if not stratify_by:
            return sorted(rng.sample(items, sample), key=lambda i: i.id)
        strata: dict[str, list[QAItem]] = {}
        for item in items:
            strata.setdefault(getattr(item, stratify_by), []).append(item)
        n_strata = len(strata)
        base_quota = sample // n_strata
        remainder = sample % n_strata
        selected: list[QAItem] = []
        for i, (_, bucket) in enumerate(sorted(strata.items())):
            quota = min(base_quota + (1 if i < remainder else 0), len(bucket))
            selected.extend(rng.sample(bucket, quota))
        return sorted(selected, key=lambda i: i.id)


def _default_fetcher(cache_dir: Path | str | None) -> Fetcher:
    from memarena.datasets._hf_fetch import HuggingFaceFetcher  # local import: network deps only when needed
    return HuggingFaceFetcher(revision=REVISION, cache_dir=Path(cache_dir) if cache_dir else Path(".cache/longmemeval_v2"))
```

Note: `_stratified_sample` duplicates `SmokeDatasetLoader`'s logic. This is an intentional, documented DRY exception for Day 2 — do not extract a shared helper under time pressure; **if touching this file again on Day 3+, extract `stratified_sample(items, sample, seed, stratify_by, key_fn)` into `datasets/base.py`** and have both loaders call it.

- [ ] **Step 4: Implement `_hf_fetch.py`** (the real, network-touching fetcher — separate module so it's trivially excluded from the unit-test import graph)

```python
# src/memarena/datasets/_hf_fetch.py
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import httpx

HF_BASE = "https://huggingface.co/datasets/xiaowu0162/longmemeval-v2/resolve/{revision}/{path}"
CHECKSUMS_PATH = "checksums.sha256"


class ChecksumMismatchError(Exception):
    pass


class HuggingFaceFetcher:
    """Real origin fetcher for LongMemEval-V2 (§5.6: download-from-origin,
    verify sha256, cache locally). Never redistributes — only caches under
    `cache_dir`, which is gitignored."""

    def __init__(self, *, revision: str, cache_dir: Path):
        self._revision = revision
        self._cache_dir = cache_dir
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._checksums: dict[str, str] | None = None

    def _url(self, path: str) -> str:
        return HF_BASE.format(revision=self._revision, path=path)

    def _load_checksums(self) -> dict[str, str]:
        if self._checksums is None:
            resp = httpx.get(self._url(CHECKSUMS_PATH), timeout=30.0, follow_redirects=True)
            resp.raise_for_status()
            checksums = {}
            for line in resp.text.splitlines():
                if not line.strip():
                    continue
                digest, name = line.split(maxsplit=1)
                checksums[name.strip()] = digest.strip()
            self._checksums = checksums
        return self._checksums

    def _fetch_and_verify_full_file(self, remote_path: str, local_name: str) -> Path:
        local_path = self._cache_dir / local_name
        if local_path.exists():
            return local_path
        resp = httpx.get(self._url(remote_path), timeout=120.0, follow_redirects=True)
        resp.raise_for_status()
        digest = hashlib.sha256(resp.content).hexdigest()
        expected = self._load_checksums().get(remote_path)
        if expected and digest != expected:
            raise ChecksumMismatchError(f"{remote_path}: expected {expected}, got {digest}")
        local_path.write_bytes(resp.content)
        return local_path

    def fetch_questions(self) -> str:
        return str(self._fetch_and_verify_full_file("questions.jsonl", "questions.jsonl"))

    def fetch_haystack_small(self) -> str:
        return str(self._fetch_and_verify_full_file("haystacks/lme_v2_small.json", "haystack_small.json"))

    def fetch_trajectories(self, needed_ids: set[str]) -> str:
        cache_key = hashlib.sha256(",".join(sorted(needed_ids)).encode()).hexdigest()[:16]
        local_path = self._cache_dir / f"trajectories_filtered_{cache_key}.jsonl"
        if local_path.exists():
            return str(local_path)

        expected = self._load_checksums().get("trajectories.jsonl")
        hasher = hashlib.sha256()
        with httpx.stream("GET", self._url("trajectories.jsonl"), timeout=600.0, follow_redirects=True) as resp:
            resp.raise_for_status()
            buffer = b""
            with local_path.open("wb") as out:
                for chunk in resp.iter_bytes(chunk_size=1 << 20):
                    hasher.update(chunk)
                    buffer += chunk
                    *complete_lines, buffer = buffer.split(b"\n")
                    for raw_line in complete_lines:
                        if not raw_line.strip():
                            continue
                        obj = json.loads(raw_line)
                        if obj["id"] in needed_ids:
                            out.write(raw_line + b"\n")
                if buffer.strip():
                    obj = json.loads(buffer)
                    if obj["id"] in needed_ids:
                        with local_path.open("ab") as out:
                            out.write(buffer + b"\n")

        digest = hasher.hexdigest()
        if expected and digest != expected:
            local_path.unlink(missing_ok=True)
            raise ChecksumMismatchError(f"trajectories.jsonl: expected {expected}, got {digest}")
        return str(local_path)
```

- [ ] **Step 5: Run to verify it passes**

Run: `pytest tests/test_longmemeval_v2.py -v`
Expected: PASS. This test run must be **fully offline** — confirm by running with network disabled if possible, or by inspecting that `FakeFetcher` never imports `httpx`/`_hf_fetch.py`. `_hf_fetch.py` itself has no dedicated unit test (it's a thin, mostly-untestable-without-network HTTP wrapper) — its correctness is validated by the real Task 7 end-to-end run, which is the intended verification point for this file per the "don't test what you can't fake cheaply, verify it live once" tradeoff.

- [ ] **Step 6: Register in `registry.py`**

```python
from memarena.datasets.longmemeval_v2 import LongMemEvalV2Loader
# ...
DATASET_REGISTRY: dict[str, type[DatasetLoader]] = {
    "smoke": SmokeDatasetLoader,
    "longmemeval_v2": LongMemEvalV2Loader,
}
```

Add a `tests/test_registry.py` case (check the existing file's style first) asserting `get_dataset_class("longmemeval_v2") is LongMemEvalV2Loader`.

- [ ] **Step 7: Run full suite + ruff**

Run: `pytest -q && ruff check src tests`
Expected: PASS, clean.

- [ ] **Step 8: Commit**

```bash
git add src/memarena/datasets/longmemeval_v2.py src/memarena/datasets/_hf_fetch.py src/memarena/registry.py tests/test_longmemeval_v2.py tests/test_registry.py tests/fixtures/lme_v2_*.json*
git commit -m "feat: real LongMemEval-V2 loader (§5.6, §8 Day 2)"
```

---

## Task 6: Mem0 adapter

**Files:**
- Create: `src/memarena/providers/mem0_adapter.py`
- Test: `tests/test_mem0_adapter.py`
- Modify: `pyproject.toml` (add `mem0ai==2.0.11` to `dependencies`)

**Real client API (confirmed live 2026-07-01 against `mem0ai==2.0.11`, do not re-derive):**
- `MemoryClient(api_key=...)`.
- `client.add(messages, user_id=..., metadata=..., timestamp=<int unix seconds>)` — returns `{"event_id": ..., "status": "PENDING"}` **immediately**; the write is processed **asynchronously** server-side. Empirically, a single short memory took ~5s to become visible via `get_all`.
- `client.search(query, filters={"user_id": ...}, top_k=...)` — **must** pass `user_id` via `filters`, not as a top-level kwarg (raises `ValueError` otherwise — `ENTITY_PARAMS = frozenset({"run_id","agent_id","user_id","app_id"})` are rejected top-level in `search()`/`get_all()`). Returns `{"results": [{"id", "memory", "user_id", "score", "score_breakdown", "metadata", "categories", "created_at", "updated_at", ...}]}`.
- `client.get_all(filters={"user_id": ...})` — same `filters`-only rule; returns `{"count", "results": [...]}`. Used by the adapter to poll for add-completion.
- `client.delete_all(user_id=...)` — accepts `user_id` as a top-level kwarg (unlike `search`/`get_all`); async, returns `{"message": "Delete in progress...", "event_id": ...}` immediately (do **not** assume synchronous completion here either — see Step 3 poll-after-reset note).
- Exceptions: `mem0.exceptions.MemoryError` (base) with subclasses `RateLimitError`, `AuthenticationError`, `ValidationError`, `MemoryQuotaExceededError`, `NetworkError`, `MemoryNotFoundError`. All I/O methods are wrapped by the SDK's own `@api_error_handler`, which raises these on `httpx.HTTPStatusError`/`httpx.RequestError`.

**Design (the one non-obvious piece — write this comment in the code, not just here):** `MemoryProvider.add()` must be a true synchronous façade (Appendix A). Because Mem0's write path is async, the adapter's `add()` polls `get_all()` for this namespace after calling `client.add()`, until the observed memory count increases (i.e., at least one new memory landed) or a timeout is hit. This means `add_latency_ms` measured by the runner will include real multi-second settle time — that is correct, not a bug: it is the true latency a synchronous caller experiences before a subsequent `search()` is reliable.

**Interfaces:**
- Consumes: `MemoryProvider`, `MemoryRecord`, `ProviderInfo` from `providers/base.py`; `ProviderError` from `errors.py`.
- Produces: `Mem0Provider(MemoryProvider)`, constructor `(config: dict, *, client: Mem0ClientProtocol | None = None)` — `client` injectable for tests (mirrors `baseline_rag`'s `embed_fn` pattern).

- [ ] **Step 1: Write the failing tests**

```python
import pytest

from memarena.errors import ProviderError
from memarena.providers.mem0_adapter import Mem0Provider


class FakeMem0Client:
    """Deterministic stand-in for mem0.MemoryClient — models the real
    async-add-then-poll behavior without any network or sleep."""

    def __init__(self):
        self.deleted: list[str] = []
        self._store: dict[str, list[dict]] = {}
        self.add_calls: list[dict] = []
        self._next_id = 0

    def add(self, messages, *, user_id, metadata=None, timestamp=None):
        self.add_calls.append({"messages": messages, "user_id": user_id})
        bucket = self._store.setdefault(user_id, [])
        for m in messages:
            self._next_id += 1
            bucket.append({"id": f"mem-{self._next_id}", "memory": m["content"], "score": None})
        return {"event_id": "evt-1", "status": "PENDING"}

    def get_all(self, *, filters):
        user_id = filters["user_id"]
        return {"results": self._store.get(user_id, [])}

    def search(self, query, *, filters, top_k=5):
        user_id = filters["user_id"]
        matches = [m for m in self._store.get(user_id, []) if query.lower() in m["memory"].lower()]
        for i, m in enumerate(matches):
            m["score"] = 1.0 - i * 0.1
        return {"results": matches[:top_k]}

    def delete_all(self, *, user_id):
        self.deleted.append(user_id)
        self._store[user_id] = []
        return {"message": "ok"}


class FailingAddClient(FakeMem0Client):
    def add(self, messages, *, user_id, metadata=None, timestamp=None):
        from mem0.exceptions import RateLimitError
        raise RateLimitError(message="slow down", error_code="RATE_001")


class TestMem0Provider:
    def setup_method(self):
        self.client = FakeMem0Client()
        self.provider = Mem0Provider({"top_k": 5}, client=self.client)

    def test_info(self):
        info = self.provider.info()
        assert info.name == "mem0"
        assert info.pricing_model == "self_hosted"  # free-tier hosted API — see LICENSES.md
        assert len(info.config_digest) == 64

    def test_reset_calls_delete_all(self):
        self.provider.reset("ns1")
        assert self.client.deleted == ["ns1"]

    def test_add_then_search_returns_relevant_record(self):
        self.provider.reset("ns1")
        self.provider.add("ns1", [{"role": "user", "content": "My dog's name is Biscuit."}],
                           session_id="s1", timestamp="2026-01-01T00:00:00Z")
        results = self.provider.search("ns1", "dog", top_k=5)
        assert len(results) == 1
        assert "Biscuit" in results[0].content
        assert results[0].score is not None

    def test_add_passes_user_id_and_messages_through(self):
        self.provider.reset("ns1")
        self.provider.add("ns1", [{"role": "user", "content": "hi"}], session_id="s1", timestamp="2026-01-01T00:00:00Z")
        assert self.client.add_calls[0]["user_id"] == "ns1"

    def test_search_top_k_is_respected(self):
        self.provider.reset("ns1")
        for i in range(10):
            self.provider.add("ns1", [{"role": "user", "content": f"fact number {i}"}],
                               session_id=f"s{i}", timestamp="2026-01-01T00:00:00Z")
        results = self.provider.search("ns1", "fact", top_k=3)
        assert len(results) == 3

    def test_client_error_raises_provider_error(self):
        provider = Mem0Provider({"top_k": 5}, client=FailingAddClient())
        provider.reset("ns1")
        with pytest.raises(ProviderError):
            provider.add("ns1", [{"role": "user", "content": "hi"}], session_id="s1", timestamp="2026-01-01T00:00:00Z")
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_mem0_adapter.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

```python
# src/memarena/providers/mem0_adapter.py
from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from typing import Any, Protocol

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from memarena.errors import ProviderError
from memarena.providers.base import MemoryProvider, MemoryRecord, ProviderInfo

CLIENT_VERSION = "2.0.11"  # pinned — mem0ai==2.0.11 in pyproject.toml
POLL_INTERVAL_S = 1.0
POLL_TIMEOUT_S = 30.0  # empirically ~5s to settle (2026-07-01); generous margin


class Mem0ClientProtocol(Protocol):
    def add(self, messages, *, user_id: str, metadata: dict | None = None, timestamp: int | None = None) -> dict: ...
    def get_all(self, *, filters: dict) -> dict: ...
    def search(self, query: str, *, filters: dict, top_k: int = 5) -> dict: ...
    def delete_all(self, *, user_id: str) -> dict: ...


def _iso_to_unix(timestamp: str) -> int:
    return int(datetime.fromisoformat(timestamp.replace("Z", "+00:00")).timestamp())


def _default_client(api_key: str | None) -> Mem0ClientProtocol:
    from mem0 import MemoryClient
    import os
    key = api_key or os.environ.get("MEM0_API_KEY")
    if not key:
        raise ProviderError("MEM0_API_KEY is not set; mem0 adapter needs it (set it in .env, never hardcode it).")
    return MemoryClient(api_key=key)


def _wrap_mem0_errors(fn):
    def wrapped(*args, **kwargs):
        from mem0.exceptions import MemoryError as Mem0MemoryError
        try:
            return fn(*args, **kwargs)
        except Mem0MemoryError as exc:
            raise ProviderError(f"mem0 client error [{exc.error_code}]: {exc.message}") from exc
    return wrapped


class Mem0Provider(MemoryProvider):
    """mem0 adapter (§5.5). Namespace = mem0 user_id (spec's convention).

    Mem0's write path (/v3/memories/add/) is asynchronous — add() returns
    {"event_id", "status": "PENDING"} immediately server-side. To honor the
    MemoryProvider sync-façade contract (Appendix A) and to make an
    immediately-following search() reliable, add() polls get_all() for this
    namespace until the observed memory count increases, up to
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
        return ProviderInfo(name="mem0", client_version=CLIENT_VERSION, config_digest=digest, pricing_model="self_hosted")

    @_wrap_mem0_errors
    def reset(self, namespace: str) -> None:
        self._client.delete_all(user_id=namespace)

    @_wrap_mem0_errors
    def add(self, namespace: str, messages: list[dict[str, str]], *, session_id: str, timestamp: str) -> None:
        before = len(self._client.get_all(filters={"user_id": namespace}).get("results", []))
        self._client.add(
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
        response = self._client.search(query, filters={"user_id": namespace}, top_k=top_k)
        records = []
        for r in response.get("results", []):
            records.append(MemoryRecord(
                id=r["id"], content=r["memory"], metadata=r.get("metadata") or {},
                score=r.get("score"), created_at=r.get("created_at"),
            ))
        return records
```

Add retry-on-rate-limit as a thin wrapper using `tenacity` around just the `add`/`search` client calls if `RateLimitError` is raised — keep it simple, one decorator reused:

```python
_retry_rate_limit = retry(
    retry=retry_if_exception_type(tuple()),  # populated below, see note
    stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=10),
)
```

Note for the implementer: import `from mem0.exceptions import RateLimitError` at module load time (it's a hard dependency now via `mem0ai` in `pyproject.toml`, so this is safe — unlike the lazy imports used elsewhere in this file for cases that must stay import-optional) and set `retry_if_exception_type((RateLimitError,))` directly; apply `@_retry_rate_limit` to `add` and `search` (below `@_wrap_mem0_errors` won't work since that already converts the exception — apply retry *inside*, i.e. decorate the raw `self._client.add(...)` call site, or simply order decorators so retry sees the original `RateLimitError` before `_wrap_mem0_errors` converts it: `@_wrap_mem0_errors` outermost, `@_retry_rate_limit` innermost on the same method).

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/test_mem0_adapter.py -v`
Expected: PASS, fully offline (no real `mem0` import side effects beyond the package being installed — `FakeMem0Client` never touches `mem0.MemoryClient`).

- [ ] **Step 5: Add `mem0ai` to `pyproject.toml` and register the provider**

```toml
dependencies = [
    "typer>=0.12",
    "pydantic>=2.6",
    "httpx>=0.27",
    "tenacity>=8.2",
    "numpy>=1.26",
    "pyyaml>=6.0",
    "python-dotenv>=1.0",
    "mem0ai==2.0.11",
]
```

```python
# registry.py
from memarena.providers.mem0_adapter import Mem0Provider
PROVIDER_REGISTRY: dict[str, type[MemoryProvider]] = {
    "baseline_rag": BaselineRAGProvider,
    "mem0": Mem0Provider,
}
```

Extend `tests/test_registry.py` with `get_provider_class("mem0") is Mem0Provider`.

- [ ] **Step 6: `pip install -e .` to pick up the new dependency in the venv, then run full suite + ruff**

Run: `pip install -e . -q && pytest -q && ruff check src tests`
Expected: PASS, clean.

- [ ] **Step 7: Commit**

```bash
git add src/memarena/providers/mem0_adapter.py src/memarena/registry.py pyproject.toml tests/test_mem0_adapter.py tests/test_registry.py
git commit -m "feat: mem0 adapter with async-write polling (§5.5, §8 Day 2)"
```

---

## Task 7: Answering layer (constant reader, abstention-aware)

**Files:**
- Create: `src/memarena/answering.py`
- Test: `tests/test_answering.py`

**Scope decision:** built and unit-tested as a standalone module per the Day 2 goal text. **Not wired into the CLI/runner's V2 comparison run** — the Day 2 exit criterion is explicitly "deterministic metrics only" (§8), and grading V2 answers needs `eval_function`-based checking (partly LLM-judge-based), which is Day 4 scope. Wiring this in prematurely would be scope creep past what Day 2 asks for. Day 4 imports and calls `answer_question` directly.

**Interfaces:**
- Produces: `ReaderAnswer` dataclass (`text: str`, `abstained: bool`), `answer_question(question: str, retrieved_contents: list[str], *, model: str = READER_MODEL, chat_fn: ChatFn | None = None) -> ReaderAnswer`, module constant `READER_MODEL = "gpt-5-mini-2025-08-07"` (pinned — confirmed available 2026-07-01), `ABSTENTION_MARKER = "I don't know"`.

- [ ] **Step 1: Write the failing tests**

```python
import pytest

from memarena.answering import ABSTENTION_MARKER, ReaderAnswer, answer_question
from memarena.errors import ProviderError


class TestAnswerQuestion:
    def test_answers_using_retrieved_context(self):
        def chat_fn(system_prompt, user_prompt):
            assert "Biscuit" in user_prompt
            return "Biscuit"

        result = answer_question("What is my dog's name?", ["My dog's name is Biscuit."], chat_fn=chat_fn)
        assert isinstance(result, ReaderAnswer)
        assert result.text == "Biscuit"
        assert result.abstained is False

    def test_abstention_marker_sets_abstained_true(self):
        def chat_fn(system_prompt, user_prompt):
            return ABSTENTION_MARKER

        result = answer_question("What is my blood type?", [], chat_fn=chat_fn)
        assert result.abstained is True
        assert result.text == ABSTENTION_MARKER

    def test_empty_retrieved_context_still_calls_reader(self):
        calls = []

        def chat_fn(system_prompt, user_prompt):
            calls.append(user_prompt)
            return ABSTENTION_MARKER

        answer_question("anything?", [], chat_fn=chat_fn)
        assert len(calls) == 1

    def test_chat_fn_failure_raises_provider_error(self):
        def bad_chat_fn(system_prompt, user_prompt):
            raise RuntimeError("network down")

        with pytest.raises(ProviderError):
            answer_question("q?", ["ctx"], chat_fn=bad_chat_fn)

    def test_system_prompt_is_abstention_aware(self):
        captured = {}

        def chat_fn(system_prompt, user_prompt):
            captured["system"] = system_prompt
            return "42"

        answer_question("q?", ["ctx"], chat_fn=chat_fn)
        assert ABSTENTION_MARKER in captured["system"]
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_answering.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

```python
# src/memarena/answering.py
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from memarena.errors import ProviderError

READER_MODEL = "gpt-5-mini-2025-08-07"  # pinned constant reader (§5.0.1) — confirmed available 2026-07-01
ABSTENTION_MARKER = "I don't know"

ChatFn = Callable[[str, str], str]  # (system_prompt, user_prompt) -> reply text

SYSTEM_PROMPT = (
    "You are answering a question using ONLY the context provided below. "
    "Do not use outside knowledge. If the context does not contain enough "
    f"information to answer confidently, reply with exactly: {ABSTENTION_MARKER}"
)


@dataclass(frozen=True)
class ReaderAnswer:
    text: str
    abstained: bool


def _default_chat_fn(model: str) -> ChatFn:
    def chat(system_prompt: str, user_prompt: str) -> str:
        import os

        import httpx
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ProviderError("OPENAI_API_KEY is not set; the answering layer needs it.")
        response = httpx.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": model,
                "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                "temperature": 0,
            },
            timeout=30.0,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()

    return chat


def answer_question(question: str, retrieved_contents: list[str], *,
                     model: str = READER_MODEL, chat_fn: ChatFn | None = None) -> ReaderAnswer:
    """Constant-reader answering layer (§5.0.1, §8 Day 2): ONE pinned reader
    model, abstention-aware prompt. Not wired into the Day-2 CLI run — see
    the Day 2 plan's Task 7 scope note; Day 4 calls this against the judge."""
    fn = chat_fn or _default_chat_fn(model)
    context = "\n".join(f"- {c}" for c in retrieved_contents) if retrieved_contents else "(no context retrieved)"
    user_prompt = f"Context:\n{context}\n\nQuestion: {question}"
    try:
        text = fn(SYSTEM_PROMPT, user_prompt)
    except ProviderError:
        raise
    except Exception as exc:
        raise ProviderError(f"answering layer reader call failed: {exc}") from exc
    return ReaderAnswer(text=text, abstained=text.strip() == ABSTENTION_MARKER)
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/test_answering.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/memarena/answering.py tests/test_answering.py
git commit -m "feat: constant-reader answering layer, abstention-aware (§5.0.1, §8 Day 2)"
```

---

## Task 8: Configs for the real Day-2 run

**Files:**
- Create: `configs/providers/mem0.default.yaml`
- Modify: `configs/pricing.yaml`
- Create: `configs/run.day2_v2.yaml`

- [ ] **Step 1: `configs/providers/mem0.default.yaml`**

```yaml
# mem0 quickstart default (§5.5). Free/Hobby tier — see datasets/LICENSES.md
# for the quota arithmetic that keeps Day 2's run inside it.
top_k: 5
```

- [ ] **Step 2: add a `mem0` entry to `configs/pricing.yaml`**

```yaml
mem0:
  pricing_model: self_hosted
  usd_per_1k_tokens: 0.0  # free/Hobby tier — no per-token charge to us; quota-limited, not cost-limited
```

- [ ] **Step 3: `configs/run.day2_v2.yaml`**

```yaml
run:
  id: day2-v2-baseline-vs-mem0
  seed: 42
  repetitions: 1
  budget_usd_max: 10.0

datasets:
  - name: longmemeval_v2
    sample: 100
    stratify_by: question_type

providers:
  - adapter: baseline_rag
    config: configs/providers/baseline.yaml
  - adapter: mem0
    config: configs/providers/mem0.default.yaml

output:
  dir: results/day2-v2-baseline-vs-mem0
```

- [ ] **Step 4: Commit**

```bash
git add configs/providers/mem0.default.yaml configs/pricing.yaml configs/run.day2_v2.yaml
git commit -m "config: Day 2 baseline-vs-mem0 run on LongMemEval-V2 (100-item sample)"
```

---

## Task 9: Execute the real end-to-end run

**Not a code task — this is the live verification step.** No test file; this is what proves the Day-2 exit criterion.

- [ ] **Step 1: Sanity-check secrets are present**

Run: `python3 -c "from dotenv import dotenv_values; v=dotenv_values('.env'); print('OPENAI_API_KEY' in v and bool(v['OPENAI_API_KEY']), 'MEM0_API_KEY' in v and bool(v['MEM0_API_KEY']))"`
Expected: `True True`

- [ ] **Step 2: Run it**

Run: `memarena run --config configs/run.day2_v2.yaml` (first invocation triggers the real ~4-5 min `trajectories.jsonl` streaming download+verify+filter described in Task 5 — this is expected, one-time, cached under `.cache/longmemeval_v2/` afterward)

Expected: two `=== ... ===` blocks (`baseline_rag on longmemeval_v2`, `mem0 on longmemeval_v2`), each printing `Recall@5: N/A`, `MRR: N/A` (per Task 1/2's documented finding — this is correct, not a bug), real `Add latency p50/p95`, real `Search latency p50/p95`, `Cost: $X.XXXX (budget: not truncated)`, and `Items: N scored (M infra errors)`.

- [ ] **Step 3: Verify the budget was respected**

Check the printed `Cost:` lines for both providers sum to ≤ $10.00, and `budget: not truncated` (or, if truncated, that it stopped cleanly with partial results, not an error).

- [ ] **Step 4: Verify quota safety**

Confirm (from the run's own behavior — no infra errors from mem0 due to rate limiting/quota) that the ~24-add/~200-search budget from Task 1's LICENSES.md arithmetic held in practice.

- [ ] **Step 5: Inspect the journal files for a sanity spot-check**

Run: `python3 -c "import json; lines=[json.loads(l) for l in open('results/day2-v2-baseline-vs-mem0/mem0__longmemeval_v2__journal.jsonl')]; print(len(lines), sum(1 for l in lines if l.get('ingested')), lines[0])"`
Expected: `ingested=True` count equals at most `2 * TRAJECTORIES_PER_DOMAIN`-worth of distinct namespace-first-hits — i.e. at most 2 (one per domain present in the sample), not 100. This is the ingestion-cache payoff — confirm it, since it's the difference between a 5-minute run and a 2-hour one.

- [ ] **Step 6: Do not commit `results/` — confirm it's gitignored**

Run: `git status --short` — `results/` must not appear (already in `.gitignore` from Day 1).

---

## Task 10: Final verification pass

- [ ] **Step 1: Full test suite**

Run: `pytest -q`
Expected: all green, zero network calls made (no test imports `_hf_fetch.py`'s real HTTP path or the real `mem0.MemoryClient`).

- [ ] **Step 2: Lint**

Run: `ruff check src tests`
Expected: clean. Fix anything flagged (import order, unused imports from the lazy-import patterns in `mem0_adapter.py`/`longmemeval_v2.py` — those are intentional but must still satisfy `ruff`'s rules; if `ruff` flags the local imports, add a narrow `# noqa` only if truly necessary and document why inline).

- [ ] **Step 3: Confirm no secrets are staged**

Run: `git status --short` and `git diff --cached -- .env` (should be empty/untracked) — `.env` must never be committed; only `.env.example` changes.

- [ ] **Step 4: Push**

```bash
git push origin main
```

(Only after explicit confirmation this matches the user's `/goal` completion criterion — `pushed to main` is explicit in the goal text, so this is expected, not a judgment call needing separate sign-off.)
