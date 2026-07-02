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
