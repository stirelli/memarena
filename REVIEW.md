# Adversarial correctness review, Days 1-2

Date: 2026-07-02. Scope: everything shipped through Day 2 (metrics, runner,
ingestion cache, stratified samplers, baseline_rag and mem0 adapters,
LongMemEval-V2 loader, CLI reporting). Method: line-by-line adversarial read
against the spec (`../memarena_spec.md`), hand-computed metric fixtures
written before looking at outputs, and a regression test for every confirmed
bug (written first, watched fail, then fixed). No features were added.

Severity scale: **High** (published numbers are wrong or misleading),
**Medium** (wrong numbers under realistic conditions), **Low** (latent risk,
documented), **Info** (worth knowing, no action needed now).

## Summary

| # | Severity | Finding | Status |
|---|----------|---------|--------|
| F1 | High | Stratified sampler used equal quotas per stratum, not proportional | Fixed |
| F2 | High | Sampler silently returned fewer items than requested | Fixed |
| F3 | Medium | Add-latency window included `reset()` | Fixed |
| F4 | Medium | mem0 settle poll required the memory count to increase | Fixed |
| F5 | Medium | Empty runs reported a fabricated 0.0 search latency | Fixed |
| F6 | Medium | Recall@k journaled for k above the actual retrieval depth | Fixed |
| F7 | Low | Dataset digest covers questions only, not trajectories or scope caps | Documented |
| F8 | Low | mem0 user_id is shared across provider configs | Documented |
| F9 | Low | Settle race if mem0 `delete_all` is itself async | Documented |
| F10 | Low | Fetcher soft-verifies checksums; local cache is trusted | Documented |
| F11 | Low | Abstention detection is exact-match on the marker string | Documented |
| F12 | Info | `difflib` autojunk can weaken fuzzy matching on long strings | Documented |
| F13 | Info | `run.concurrency` config field is parsed but unused | Documented |

All 6 fixes carry regression tests. Suite: 134 passed, ruff clean.

## Fixed

### F1 (High): sampler was equal-quota, not proportional

`_stratified_sample` (duplicated in `smoke.py` and `longmemeval_v2.py`)
allocated `sample // n_strata` per stratum plus alphabetical remainder.
LongMemEval-V2 strata are heavily skewed (134 static-environment vs 29
errors-gotchas out of 451), so the Day 2 run sampled errors-gotchas at 48%
and static-environment at 10%. The aggregate metric of such a sample does
not estimate the dataset. Verified against the real cached dataset:

- Day 2 journal composition (old): 15/15/14/14/14/14/14 by question type.
- Proportional composition (new): 30/19/16/12/9/7/7.

Fix: one shared `datasets/sampling.py` with largest-remainder (Hamilton)
allocation, seeded, remainder ties broken by stratum name. Both loaders now
delegate to it; the two divergent copies are gone.
Tests: `tests/test_sampling.py` (hand-computed quotas for three corpora,
determinism, sortedness).

### F2 (High): silent sample shortfall

The old code capped each quota at the bucket size with no redistribution:
with strata {a:1, b:9} and `sample=6` it returned 4 items and nobody would
know. On the real dataset, `sample=250` would have silently returned 239.
Largest-remainder allocation provably never exceeds a stratum when
`sample < N`, so the result now always has exactly `sample` items.
Test: `TestNoSilentShortfall` in `tests/test_sampling.py` (fails on the old
implementation with 4 != 6).

### F3 (Medium): `reset()` was inside the add-latency window

Spec Appendix A and section 5.7 define latency as wall-clock around
`add`/`search`, measured by the runner. `runner._ingest` timed
`reset() + add()*sessions` together. For mem0, `reset()` is a `delete_all`
API call, so the published add latencies (51.9s and 79.3s per domain in the
Day 2 journals) include deletion time. `reset()` now runs before the timer
starts. Test: `TestLatencyWindows` in `tests/test_runner.py` drives the
runner with a fake clock (reset 5s, add 1s, search 0.5s) and asserts
add_latency_ms == 1000.0 exactly; the old code reported 6000.0. The same
test pins the search window to `provider.search` only (500.0ms).

### F4 (Medium): mem0 settle detection required a count increase

`_poll_until_visible` polled `get_all()` until the memory count exceeded
the pre-add count. mem0's own extraction pipeline resolves updates (the
adapter even declares `supports_update_resolution = True`): an add can
consolidate into an existing memory (UPDATE) or remove one (DELETE), leaving
the count flat or lower. Any such write would stall the full 30s timeout and
surface as a false `infra_error`, biasing the leaderboard against mem0 on
exactly the datasets with redundant content (LongMemEval-V2 trajectories are
highly repetitive). The Day 2 run happened to only trigger count-increasing
adds, so this never fired live, but it is wrong by mem0's documented
semantics.

Fix: settle is now detected as ANY change to the namespace snapshot
(frozenset of `(id, memory, updated_at)`), covering ADD, UPDATE and DELETE
resolutions. A true no-op write still polls to the deadline and raises
loudly (see Known limitations). Tests: `TestSettleDetection` in
`tests/test_mem0_adapter.py` with a consolidating fake client (old code
times out) and a no-op client (still raises).

### F5 (Medium): fabricated 0.0 latency for empty runs

`percentile([])` returns 0.0, and `RunMetrics.search_latency_p50/p95` were
non-optional floats. A run where every item infra-errored printed
"Search latency p50/p95 (ms): 0.0 / 0.0", which reads as an instant run,
not an empty one. Search percentiles are now `float | None` via
`percentile_of_defined`, and the CLI prints N/A. Tests:
`TestEmptyRunHasNoFabricatedNumbers` in `tests/test_metrics_fixtures.py`
and `TestPrintResultNASafety` in `tests/test_cli.py`.

### F6 (Medium): Recall@k journaled beyond the retrieval depth

The runner computed Recall@k for k in (1, 3, 5, 10) regardless of `top_k`.
With the default `top_k=5`, the journals recorded a "recall_at_k: 10" that
was structurally identical to Recall@5: the measurement never requested 10
results, so labeling it @10 misrepresents the condition. The runner now
computes only `k <= top_k` and threads the same `k_values` through item
metrics, aggregation and the journal. Test: `TestRecallKsCappedAtTopK` in
`tests/test_runner.py`. Note that k above the number of records a provider
happened to return stays legitimate (a store can simply be small); the cap
applies only to the requested depth.

## Add-latency semantics (goal item 5, now explicit)

**add_latency_ms measures time-to-settled, per item, excluding reset.**
Precisely: wall-clock from the first `add()` call of an item's sessions to
the return of the last one, where every adapter's `add()` must return only
once the write is queryable.

- baseline_rag: writes are synchronous in-memory plus the embedding API
  call, so settled-on-return holds trivially.
- mem0: `/v3/memories/add/` is async (returns `{"event_id", "PENDING"}`).
  The adapter polls `get_all()` until the namespace snapshot changes, so
  its add latency includes the settle time and the polling get_all calls.
  That is the real cost of making an async write queryable, not a
  measurement artifact, and it is the same quantity (time-to-settled) the
  synchronous provider reports.

This is documented in `providers/base.py` (contract), `mem0_adapter.py`
(mechanism) and `runner._add_sessions` (measurement window). Items that
reuse a cached ingestion have `add_latency_ms = None` and are excluded from
percentiles, never counted as 0.0.

## Verified correct (no change needed)

- **Recall@k, MRR, gold-evidence matching**: five hand-computed fixtures in
  `tests/test_metrics_fixtures.py` assert exact values for hit-at-rank-3,
  k larger than the result count, abstention exclusion (None, excluded from
  means, never 0.0), duplicate retrieved records (they occupy real ranks),
  duplicate gold entries (still binary), multi-gold rank resolution, and a
  full aggregate with hand-derived numpy linear percentiles (p95 of
  [10,30,50] = 48.0, and so on). All passed on first run against the
  existing implementation.
- **Cost meter**: `estimate_cost_usd` verified against a hand-computed
  token count (1008 chars = 252 tokens at the documented 4 chars/token =
  $0.252 at $1.00/1k), plus cached-ingestion accounting (second item in a
  shared namespace pays only its question). The chars/4 approximation is
  documented in `configs/pricing.yaml` and the runner.
- **Search latency window**: wraps `provider.search` only (exact 500.0ms
  assertion under the fake clock).
- **Ingestion cache key**: includes provider name, client_version,
  config_digest, dataset_digest and namespace; each component now has an
  inequality test in `tests/test_cache.py`.
- **Namespace isolation across providers**: baseline_rag stores memories in
  a per-instance dict, mem0 in its external store keyed by
  `user_id = namespace`; the two cannot see each other's data. Within the
  shared-domain design, the runner resets a namespace before its first
  ingestion of every run (the ingestion cache is per-process and starts
  empty), so a mem0 run never searches another config's or another run's
  leftovers. The CLI runs providers sequentially with a fresh cache per
  provider x dataset pair.
- **N/A propagation**: item-level None -> journal `null` -> aggregation
  excludes None (`mean_of_defined`, `percentile_of_defined`) -> CLI prints
  N/A. No leaderboard sorting exists yet (Day 3+); when it lands, None must
  sort as absent, not as 0.

## Documented, not fixed (no confirmed harm today)

- **F7**: `dataset_digest` is the sha256 of `questions.jsonl` content only.
  Trajectory content and the ingestion scope caps (TRAJECTORIES_PER_DOMAIN,
  STATES_PER_TRAJECTORY, ACCESSIBILITY_TREE_CHARS) are not part of it.
  Harmless while the ingestion cache is in-process (caps are constants
  within a process), but the planned persistent sqlite cache (spec 5.2)
  must fold them in before it ships, or cap changes will silently reuse
  stale ingestions.
- **F8**: mem0 `user_id` is the bare namespace, shared by every mem0 config
  and every run against the same account. Safe today because runs are
  sequential and reset-before-first-ingest guarantees a clean namespace.
  Two concurrent runs would corrupt each other. When Day 3 concurrency
  lands, prefix the user_id with a short config/run digest.
- **F9**: if mem0's `delete_all` is eventually asynchronous, deletions
  settling during an add-poll could change the snapshot before the added
  memory is visible (early settle), or under the old count logic, time out.
  Live evidence (Day 2 run: zero infra errors) says deletes settle fast;
  a definitive fix needs mem0's event-status API.
- **F10**: `_hf_fetch` verifies sha256 only when the artifact appears in
  the origin's `checksums.sha256` (`if expected and ...`), and once a file
  is in the local cache it is trusted without re-hashing. Acceptable for a
  pinned-revision origin; tighten before third parties reproduce runs.
- **F11**: `answering.py` counts an answer as abstention only if it equals
  "I don't know" exactly after strip; "I don't know." (trailing period)
  would count as an answer. The module is not wired into Day 2 runs;
  address it with the Day 4 judge work, ideally at grading time rather
  than string-match time.
- **F12**: `content_matches` uses `SequenceMatcher` with default autojunk;
  for strings over 200 chars, frequent characters get junk-classified and
  the ratio can drop unexpectedly. Today's gold evidence strings are short
  sentences, so no impact; revisit if long evidence spans appear.
- **F13**: `RunSection.concurrency` is parsed and ignored (runner is
  sequential). Fine for Day 2; remove or implement in Day 3 so configs do
  not promise what the harness does not do.

## Known limitation kept on purpose

A mem0 add whose extraction resolves to a true no-op (nothing new, nothing
changed) is indistinguishable from a lost write through `get_all()` alone.
The adapter polls to the deadline and raises ProviderError; the runner
records a visible `infra_error`. This is conservative and loud rather than
silently optimistic. The clean fix is polling mem0's event status endpoint
instead of inferring from `get_all()`; that is an API-surface change, out
of scope for this review.

## Impact on existing artifacts

`results/day2-v2-baseline-vs-mem0/` was produced before these fixes: its
sample composition is the equal-quota one (F1) and its two add-latency
values include reset time (F3). The journals remain valid as raw records of
what ran, but the run should be re-executed with the fixed harness before
any number from it is published or compared. Retrieval metrics were N/A for
this dataset either way (no evidence spans), so no published accuracy
number is affected.
