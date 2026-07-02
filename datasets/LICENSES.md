# Dataset & Provider License Audit (§5.6)

Verified 2026-07-01, extended 2026-07-02 (Day 3). Record updated whenever a
new dataset/provider is added.

## LongMemEval (V1): PRIMARY dataset (§8 Day 3, revised)

- Code: https://github.com/xiaowu0162/LongMemEval, **MIT** (verified in-repo
  2026-07-02; the paper's stated intent matches the actual LICENSE file).
- Data: https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned,
  dataset card declares **MIT** (`license:mit` tag, verified 2026-07-02).
  Pinned revision `98d7416c24c778c2fee6e6f3006e7a073259d48f` (last modified
  2025-09-19). Hosting is standard Hugging Face hub terms; no gating, no
  extra usage restrictions on the card.
- The original HF dataset (`xiaowu0162/longmemeval`) is **deprecated** by the
  authors in favor of `longmemeval-cleaned`, which removes noisy history
  sessions that interfere with answer correctness. We use the cleaned
  release, artifact `longmemeval_s_cleaned.json` (277,383,467 bytes),
  sha256 `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442`
  (matches the origin's LFS digest). The loader hard-verifies this digest on
  download AND re-hashes the local cache on every load; a mismatch is a
  refusal, not a warning (closes review finding F10 for this dataset).
- Redistribution: none. Download-from-origin only; this repo ships zero
  dataset bytes.
- Evidence labels: `answer_session_ids` plus per-turn `has_answer` flags.
  These map to `gold_evidence`, which is why V1 is the PRIMARY dataset:
  Recall@k, NDCG@k and MRR are real here. V2 remains the latency/cost-only
  secondary track (see below). The 30 abstention items (`_abs` id suffix)
  form the 7th sampling stratum, carry no gold evidence, and are excluded
  from retrieval metrics (None, never 0.0).
- Scope cap for free-tier runs (documented, never silent): each item ingests
  all evidence sessions plus earliest distractors up to 8 sessions, against
  the official ~48-session haystack. A capped haystack makes retrieval
  strictly EASIER, so capped numbers are not comparable with the paper's
  full-haystack numbers and are labeled accordingly. Full-haystack runs are
  the published-batch protocol (and what Ring-2 engagements pay for). The
  cap exists because 200 items x 48 sessions x 4 providers would exceed
  Mem0's free-tier add quota (9,600 adds vs 10,000/month) on one run.

## LongMemEval-V2: secondary track (latency/cost only)

- Code: https://github.com/xiaowu0162/LongMemEval-V2, **Apache-2.0**.
- Data: https://huggingface.co/datasets/xiaowu0162/longmemeval-v2, **Apache-2.0**,
  pinned revision `f152293e235517d504809563c833d7190b8c713b`.
- Coverage-map check (Day 3, goal step 5): the release was re-inspected on
  2026-07-02 (full HF file list + repo docs). It does **not** include
  question-to-answer-bearing-trajectory coverage maps; only the shared
  per-domain haystack files exist. Trajectory-level recall for V2 therefore
  stays blocked on upstream labels; re-check on the next V2 revision.
- Redistribution: none. The loader downloads from origin at the pinned revision and
  verifies sha256 against the dataset's own `checksums.sha256`; this repo ships zero
  dataset bytes.
- Schema is **not** chat conversation data. It is 451 questions graded against
  1,870 web-agent trajectories (WebArena/ServiceNow-style), with two shared
  per-domain "haystacks" of candidate trajectories. There is no evidence-span
  field. See `src/memarena/datasets/longmemeval_v2.py` module docstring for the
  full field-level mapping this harness uses.
- Consequence for metrics: Recall@k / MRR (Level-1 retrieval metrics, §5.7) are
  **not computable** for this dataset without evidence spans that do not exist
  in the source data; we report them as N/A rather than fabricate an
  evidence-matching heuristic. Latency, cost, and ingestion throughput are real
  and fully reported. Answer-correctness grading (`eval_function`-based, partly
  deterministic / partly LLM-judge-based) is scoped to Day 4.

## LoCoMo

- https://github.com/snap-research/locomo: **CC BY-NC 4.0** (non-commercial).
  Confirmed via `LICENSE.txt`, matches the spec's prediction (§5.6).
- Per spec R1: running evaluations and publishing aggregate *scores* is
  defensible under this license; redistributing the underlying conversations
  is not, and we never do that for any dataset regardless of license.
- **No loader implemented as of Day 2.** Deferred; not on the Day-2 build list.

## MemoryAgentBench

- https://github.com/HUST-AI-HYZ/MemoryAgentBench: **MIT**. Fully permissive.
- **No loader implemented as of Day 2.** Deferred; not on the Day-2 build list.

## Mem0 (provider, not a dataset)

- Free ("Hobby") tier, confirmed via https://mem0.ai/pricing and the live
  account (`client.get_project()`, project `default-project`,
  2026-07-01): **10,000 memory-add requests/month, 1,000 retrieval calls/month,
  1 project.**
- Day 2's real run against LongMemEval-V2 (100-item stratified sample, 2
  providers) issues at most ~24 `add()` calls (2 domains × up to 6 trajectories
  × 2 providers, since ingestion is cached per-namespace; see
  `src/memarena/cache.py`) and ~200 `search()` calls (100 items × 2 providers).
  Both are comfortably under quota (<1% and ~20% respectively, worst case).
  Tier used: **free/Hobby**, `mem0ai==2.0.11`.
- Day 3 quota arithmetic (goal step 0, checked 2026-07-02): the account API
  exposes no usage counter, so remaining quota is reconstructed from our own
  journals. July usage before the Day 3 run: 100 search calls (Day 2 journal,
  100 ok items) plus a handful of one-off live confirmations. Estimated
  remaining search quota: ~880 of 1,000, which is above the 700 threshold,
  so the Day 3 run keeps **3 repetitions for all providers** (200 items x 3
  reps = 600 searches for mem0; ingestion is cached across repetitions,
  which the methodology notes as cached-retrieval reps per section 5.3).
- Ambiguity, recorded (and RESOLVED, see next bullet): mem0's pricing page
  does not define whether `get_all()` (used by our settle polling) counts
  toward "retrievals". Day 2's live run (~170 get_all calls) was not
  throttled or blocked.
- RESOLUTION, measured live 2026-07-02 during the Day 3 run: the platform
  free tier bills `search()` AND `get_all()` against ONE 1,000/month
  retrieval bucket. Quota errors carry {"event_type": "SEARCH"}; after the
  bucket emptied, `add()` still succeeded while both read endpoints
  returned HTTP 429 "Usage quota exceeded for this billing period". The
  Day 3 platform attempt died 21 items in (evidence journal preserved at
  `results/day3-v1-four-providers/mem0__platform_quota_blocked__journal.jsonl`).
- Consequence: the mem0 row for Day 3 runs **mem0 OSS self-hosted**
  (mem0ai==2.0.11 `Memory`, extraction LLM pinned to gpt-4.1-mini at
  temperature 0, embedded qdrant store), which is the vendor's open-source
  core with no platform quota. Cost is metered as OpenAI usage
  (configs/pricing.yaml). OSS gates timestamp backdating behind a platform
  API key (ValueError "Temporal reasoning requires a Mem0 API key",
  measured live), so the row carries supports_temporal=False as a
  leaderboard annotation. The platform config remains available at
  configs/providers/mem0.platform.yaml for paid/vendor-sponsored runs.

## Zep (provider, not a dataset)

- Free plan, confirmed via the live account dashboard (app.getzep.com,
  2026-07-02): **10,000 flex credits/month, 1 credit per episode ingested,
  2 projects; retrieval is not credit-billed. Free tier processes episodes
  on a lower-priority queue.** Measured live: ~10-27s per episode.
- Credit arithmetic that drove the adapter's ingestion path (see
  `configs/providers/zep.default.yaml`): per-turn chat-thread ingestion of
  the Day 3 sample (200 items x 8 sessions x ~10.3 turns) is ~16,500
  episodes/credits, over the 10,000/month plan; session-level `graph.add`
  text episodes are ~2,800 credits and carry identical content. Account
  had 757 credits used before the Day 3 run (probes included).
- Cloud outcome (measured live 2026-07-02, mid-run): beyond credits, the
  free plan enforces a small rolling REQUEST budget. The Day 3 cloud run
  was hard-throttled ("Rate limit exceeded for FREE plan") after ~1.9k
  requests (~3h, 31 items of rep 0), and a restarted shard with 1s
  client-side pacing plus ~6-minute patient retries got ZERO further items
  through. Evidence journals preserved at
  `results/day3-v1-four-providers/zep__free_plan_throttle_cascade__journal.jsonl`
  (first cascade) and the restarted shard's journal.
- Consequence: the zep row for Day 3 runs **Graphiti self-hosted**
  (graphiti-core==0.29.2, Zep's open-source core, extraction LLM pinned to
  gpt-4.1-mini at temperature 0, embedded kuzu graph, episode BM25+RRF
  retrieval). Two graphiti-on-kuzu upstream gaps are worked around and
  documented in the adapter: the driver never initializes `_database`
  (read by add_episode), and the FTS indexes its search needs are never
  created and are static once built (the adapter creates them and rebuilds
  lazily after ingestion). Cost is metered as OpenAI usage
  (configs/pricing.yaml). Cloud config remains at
  configs/providers/zep.cloud.yaml for paid/vendor-sponsored runs.
- Tier used: cloud attempt **free** (`zep-cloud==3.23.0`), final row
  **self-hosted** (`graphiti-core==0.29.2`).

## Letta (provider, not a dataset)

- Letta Cloud free tier: **5,000 credits/month** (credits bill model
  requests; the docs do not state that archival passage inserts/searches
  consume credits) and a hard cap of **3 agents** (verified live
  2026-07-02: agents.create returns 402 with {"limit": 3}). The 3-agent
  cap forces the adapter's shared-agent design with per-namespace tag
  isolation (see configs/providers/letta.default.yaml). Measured live:
  `passages.create` embeds synchronously (~1.8s for a 17k-char passage),
  `passages.search` ~1s.
- Day 3 usage: ~200 agent creations + 1,600 passage inserts + 600
  searches. If passage operations turn out to be credit-billed, the run
  journal will surface the quota block as infra_errors; we report, never
  mask.
- Tier used: **free**, `letta-client==1.12.1`.
