# Day 3 run results: LongMemEval V1, 200-item stratified sample

Run id `day3-v1-four-providers`, 2026-07-02. Dataset: LongMemEval V1
(`longmemeval_s_cleaned.json`, HF revision `98d7416c`, artifact sha256
`d6f21ea9d60a...`), stratified sample n=200 over the 7 official question
types, seed 42, repetitions 3 (ingestion cached across repetitions),
top_k 5, budget_usd_max 15 per provider (never tripped). Journals in this
directory are the raw records; aggregate them with
`scripts/day3_results_summary.py`.

## Run status

| Provider | Mode | Status |
|---|---|---|
| baseline_rag | local + OpenAI embeddings | complete (600 rows, 0 infra errors) |
| letta | vendor cloud (free tier) | complete (600 rows, 0 infra errors) |
| mem0 | self-hosted OSS (pivot, see below) | complete (600 rows, 0 infra errors) |
| zep | self-hosted Graphiti (pivot, see below) | PARTIAL: operator-stopped at 80/200 items of rep 0 to re-run another day (journal `zep__partial_operator_stopped__journal.jsonl`, 1 infra error: a kuzu internal error on one item) |

## Verbatim evidence retrieval (deterministic, extractive rows only)

Verbatim metrics apply to stores that return source text; abstractive
rows report N/A by policy, never 0.0 (docs/METHODOLOGY_NOTES.md). The
sequential re-pass reproduced the full run's verbatim numbers exactly, a
determinism check on the whole retrieval+matching pipeline.

| Provider | Representation | Verbatim R@5 | Verbatim NDCG@5 | Verbatim MRR | Scored rows |
|---|---|---|---|---|---|
| baseline_rag | extractive | 0.888 | 0.664 | 0.706 | 564 |
| letta | extractive | 0.995 | 0.942 | 0.953 | 564 |
| mem0 (OSS) | abstractive | N/A | N/A | N/A | 0 (by policy) |
| zep (graphiti), partial n=79 | extractive | 0.838 | 0.741 | 0.768 | 76 |

Notes: 564 = 600 rows minus the 36 abstention-stratum rows (12 items x 3
reps), which carry no gold evidence. The mem0 full-run journal predates
the verbatim rename and records 0.0 under the old keys; the re-pass
journal records the current N/A policy. The zep partial row is
indicative only (first 79 items of rep 0, biased toward nothing in
particular but not the full stratified sample). Cross-paradigm retrieval
quality is measured by the judged evidence_coverage metric (Day 4).

## Latency (per provenance)

PROVISIONAL numbers come from the full run, whose four shards ran
concurrently on one machine. PUBLISHABLE numbers come from the strictly
sequential re-pass (nothing else running).

Search latency, publishable (sequential re-pass, n=200 searches each;
baseline n=400 across its two repetitions):

| Provider | search p50 ms | search p95 ms |
|---|---|---|
| baseline_rag | 692 | 1226 |
| mem0 (OSS) | 1129 | 1752 |
| letta (cloud) | 1212 | 1744 |
| zep (graphiti) | pending its re-run | pending |

Add latency (time-to-settled semantics for all rows below; see
providers/base.py for the two documented semantics):

| Provider | add p50 ms | add p95 ms | Provenance |
|---|---|---|---|
| baseline_rag | 13451 | 19956 | publishable: sequential re-pass rep 0, n=200 items (8 sessions each) |
| mem0 (OSS) | 82222 | 106896 | indicative p50, fresh-ingest stratified subsample n=15, sequential, separate store |
| letta (cloud) | 15125 | 19790 | PROVISIONAL (concurrent period, full run n=200) |
| zep (graphiti) | 309695 (n=79) | 513140 | PROVISIONAL and partial; re-measure on its re-run |

Per-item add latency covers a full item (8 sessions). mem0's and
graphiti's numbers are dominated by their synchronous local extraction
LLM passes; letta's by 8 sequential cloud passage inserts; baseline's by
embedding calls.

## Cost

Metered (chars/4 approximation at documented blended rates,
configs/pricing.yaml): baseline $0.088 per full pass (x2 with the
re-pass), mem0-OSS $2.64 (run) + $0.20 (n=15 probe), zep-graphiti $2.61
(partial run). letta and the cloud attempts metered $0 (free tiers;
quota-limited, not cost-limited).

Reconciliation against real OpenAI usage (goal addendum): the org-level
July spend on the OpenAI dashboard read **$7.36 of a $50 cap at
2026-07-02 14:15 ART**, which bounds every OpenAI cost this month:
Days 1-2 development, all Day 3 runs and probes, the re-pass, and the
Day 4 answer generation. The API key lacks the api.usage.read scope and
the dashboard's per-model breakdown lags for same-day usage, so
row-level attribution is deferred to the zep re-run day; the metered
row totals above (~$5.6) are consistent with the real total once
development and reader-generation usage is added. Budget_usd_max 15 was
respected by every runner invocation, and total real spend sits far
under it.

## Free-tier findings (the run's own story, evidence preserved)

1. mem0 platform free tier bills search() AND get_all() against one
   1,000/month retrieval bucket; the run died 21 items in
   (`mem0__platform_quota_blocked__journal.jsonl`). Pivot: mem0 OSS
   self-hosted (vendor's own open-source core, pinned extraction LLM).
2. Zep Cloud free plan hard-throttles after a small rolling request
   budget (~1.9k requests in ~3h); a paced, patiently-retrying restart
   got zero further items (`zep__free_plan_throttle_cascade__journal.jsonl`).
   Pivot: Graphiti (Zep's open-source core) with embedded kuzu.
3. Letta Cloud's free tier sustained the full run (with the shared-agent
   design forced by its 3-agent cap). Of the three memory clouds, it is
   the only one whose free tier survived a 200-item benchmark.

## Reproduce

```bash
memarena run --config configs/run.day3_v1.yaml --provider <name>
python scripts/day3_latency_repass.py --provider baseline_rag --provider mem0 --provider letta
python scripts/day3_results_summary.py
```
