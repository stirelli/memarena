"""Day 3 sequential latency re-pass (§8 Day 3, post-run addendum).

The four Day 3 shards ran CONCURRENTLY on one machine, so their latency
percentiles are contaminated by co-scheduling and shared bandwidth; those
journals stay marked PROVISIONAL. This script produces the publishable
numbers, strictly sequentially (one provider at a time, nothing else
running):

Phase 1 — search-only re-pass, all four providers, full 200-item sample:
  the ingestion cache is pre-seeded so run() never re-ingests; providers
  attach to the stores the full run left behind (letta: the shared cloud
  agent; mem0-oss: the on-disk qdrant store; zep-graphiti: the on-disk
  kuzu store). baseline_rag is process-local, so it runs repetitions=2
  fresh: rep 0 doubles as its sequential (publishable) add-latency pass
  and rep 1 is its cached search-only pass.

Phase 2 — fresh-ingest add-latency subsample for the self-hosted rows
  (mem0-oss, zep-graphiti): a stratified n=15 subsample of the same
  200-item sample, ingested sequentially into SEPARATE throwaway stores
  (the full-run stores are preserved for Day 4). Reported as indicative
  p50 with n declared.

Run: .venv/bin/python scripts/day3_latency_repass.py
"""
from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

from memarena.cache import IngestionCache, ingestion_cache_key
from memarena.config import load_yaml_dict
from memarena.datasets.longmemeval_v1 import LongMemEvalV1Loader
from memarena.datasets.sampling import stratified_sample
from memarena.registry import get_provider_class
from memarena.runner import run

OUT_DIR = Path("results/day3-v1-four-providers")
SEED = 42
SAMPLE = 200
ADDLAT_N = 15
BUDGET_USD_MAX = 15.0

PROVIDER_CONFIGS = {
    "baseline_rag": "configs/providers/baseline.yaml",
    "mem0": "configs/providers/mem0.default.yaml",
    "zep": "configs/providers/zep.default.yaml",
    "letta": "configs/providers/letta.default.yaml",
}

# Throwaway stores for the fresh-ingest probes — never touch the
# full-run stores, which Day 4 (judge work) still needs.
ADDLAT_OVERRIDES = {
    "mem0": {"oss_vector_store_path": ".cache/mem0_oss_addlat"},
    "zep": {"graphiti_db_path": ".cache/zep_graphiti_addlat/graph.kuzu"},
}


def _load_items():
    loader = LongMemEvalV1Loader()
    items = loader.load(sample=SAMPLE, seed=SEED)
    return items, loader.sha256()


def _provider(name: str, overrides: dict | None = None):
    config = load_yaml_dict(PROVIDER_CONFIGS[name]) | (overrides or {})
    return get_provider_class(name)(config), config


def search_only_repass(items, digest, pricing) -> None:
    for name in ("baseline_rag", "mem0", "zep", "letta"):
        provider, config = _provider(name)
        journal = OUT_DIR / f"{name}__longmemeval_v1__repass__journal.jsonl"
        if name == "baseline_rag":
            repetitions, cache = 2, None  # rep 0 re-ingests (in-memory store); rep 1 is the cached pass
        else:
            repetitions = 1
            cache = IngestionCache()
            info = provider.info()
            for item in items:
                cache.mark_ingested(ingestion_cache_key(info, dataset_digest=digest, namespace=item.namespace))
        result = run(
            provider, items, run_id="day3-v1-latency-repass", seed=SEED, dataset_digest=digest,
            repetitions=repetitions, top_k=config.get("top_k", 5), budget_usd_max=BUDGET_USD_MAX,
            pricing=pricing.get(name), journal_path=journal, ingestion_cache=cache,
        )
        m = result.metrics
        print(f"[repass] {name}: search p50/p95 = {m.search_latency_p50_ms} / {m.search_latency_p95_ms} ms "
              f"({m.n_items} rows, {result.infra_error_count} infra errors)", flush=True)


def fresh_ingest_addlatency(items, digest, pricing) -> None:
    subsample = stratified_sample(items, sample=ADDLAT_N, seed=SEED, stratify_by="question_type")
    for name in ("mem0", "zep"):
        provider, config = _provider(name, ADDLAT_OVERRIDES[name])
        journal = OUT_DIR / f"{name}__longmemeval_v1__addlatency_n{ADDLAT_N}__journal.jsonl"
        result = run(
            provider, subsample, run_id=f"day3-v1-addlatency-n{ADDLAT_N}", seed=SEED, dataset_digest=digest,
            repetitions=1, top_k=config.get("top_k", 5), budget_usd_max=BUDGET_USD_MAX,
            pricing=pricing.get(name), journal_path=journal, fresh_ingest=True,
        )
        m = result.metrics
        print(f"[addlat n={ADDLAT_N}] {name}: add p50/p95 = {m.add_latency_p50_ms} / {m.add_latency_p95_ms} ms "
              f"({result.infra_error_count} infra errors)", flush=True)


def main() -> None:
    load_dotenv()
    items, digest = _load_items()
    pricing = load_yaml_dict("configs/pricing.yaml")
    print(f"sequential re-pass over {len(items)} items (digest {digest[:12]})", flush=True)
    search_only_repass(items, digest, pricing)
    fresh_ingest_addlatency(items, digest, pricing)
    print("done", flush=True)


if __name__ == "__main__":
    main()
