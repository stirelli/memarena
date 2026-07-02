# Methodology notes (pre-1.0)

Working notes that feed METHODOLOGY.md v1.0 (Day 5). Each note records a
decision, the evidence that forced it, and what it changes in the harness.

## Verbatim evidence recall, and why abstractive stores report N/A

**Decision (2026-07-02).** The Level-1 deterministic retrieval metrics are
named **verbatim evidence recall / verbatim NDCG / verbatim MRR**, and they
are only computed for providers whose search returns source text. Every
provider carries a first-class attribute `memory_representation`:

- `extractive`: search returns source text (chunks, transcripts, raw
  episodes). Verbatim matching against gold evidence is meaningful.
  Day 3 rows: baseline_rag, letta (transcript passages), zep-graphiti
  (episode scope).
- `abstractive`: search returns rewritten, distilled memories. Verbatim
  metrics are reported **N/A, never 0.0**. Day 3 rows: mem0 (both platform
  and OSS modes), zep with edge scope.

**Evidence.** In the Day 3 run, mem0's verbatim recall computed to 0.000
across every scored item. Hand inspection showed the number was false as a
claim about retrieval. For item `3fdac837` ("What is the total number of
days I spent in Japan and Chicago?"), mem0's top-5 included:

> "User visited Japan from April 15, 2023 to April 22, 2023 and fell in
> love with Tokyo during that trip."
> "User wants Italian restaurant recommendations for Chicago, noting they
> enjoyed great Italian food during a recent trip."

The information needed to answer is retrieved; it is paraphrased, so no
verbatim match against the gold evidence turns is possible. A published
0.000 would say "this system retrieves nothing", which is wrong, does not
survive a screenshot without its footnote, and hands the vendor a winnable
dispute. The same principle already governed LongMemEval-V2 (no gold
evidence labels, so retrieval metrics report N/A rather than a fabricated
number); this extends it to stores whose representation makes the labels
inapplicable.

**What verbatim metrics still measure.** For extractive stores they remain
fully comparable, deterministic, and cheap: did the store surface the
evidence text, at what rank. They stop being a cross-paradigm quality
ranking. Cross-paradigm retrieval quality is measured by the judged metric
below.

## evidence_coverage (judged, Day 4)

**Definition.** For a scored item, the judge sees the question, the gold
answer, the gold evidence, and the retrieved set (top-k contents as the
reader would see them). Binary verdict: is the information needed to answer
present in the retrieved set? Paraphrased content counts; the judge is
scoring information presence, not string overlap.

**Calibration.** Same protocol as answer correctness (spec section 5.8):
versioned judge (model id, prompt sha256, temperature 0, JSON schema),
human labels, published Cohen's kappa with the 0.75 ship gate. The Day 4
hand-labeling sheet gains **~60 binary evidence_coverage items**,
stratified by question type, half drawn from extractive rows and half from
abstractive rows, so the kappa is measured on both representations.

**Role.** evidence_coverage is the cross-paradigm retrieval-quality metric
on the leaderboard. Verbatim metrics stay as the deterministic instrument
for extractive stores (and as a free consistency check wherever both
exist).

## Failure-bucket cascade under two representations (Day 4)

The `retrieval_miss` bucket (spec section 5.9) must not silently classify
every abstractive-store failure as a retrieval miss, which is exactly what
verbatim matching would do. The cascade therefore decides retrieval_miss
per representation:

- extractive providers: verbatim matching (deterministic, as specified);
- abstractive providers: judged evidence_coverage (a failed item whose
  retrieved set covers the evidence is NOT a retrieval_miss; it falls
  through to update-conflict or synthesis buckets).

The bucket assignment records which detector decided it, so bucket
distributions remain auditable per provider.

## Latency provenance for the Day 3 run

The four Day 3 shards ran concurrently on one machine; their latency
percentiles are marked PROVISIONAL in all reporting. Publishable latencies
come from the sequential re-pass (scripts/day3_latency_repass.py): a
search-only repetition per provider run strictly one at a time, plus a
fresh-ingest n=15 stratified add-latency probe for the self-hosted rows,
reported as indicative p50 with n declared. Add-latency semantics per
provider (time-to-settled vs time-to-accepted plus settle) are defined in
providers/base.py and each adapter's docstring.

## Hosting mode is a first-class row annotation

Measured 2026-07-02: mem0's platform free tier bills search and get_all
against one 1,000/month retrieval bucket (exhausted 21 items in), and Zep
Cloud's free plan hard-throttles after a small rolling request budget
(~1.9k requests). Those rows therefore run the vendors' open-source cores
locally (mem0 OSS; Graphiti with embedded kuzu), while letta runs on the
vendor's cloud, whose free tier sustained the full run. Leaderboard rows
carry `cloud` vs `self-hosted` plus pinned versions; latency and cost are
never compared across hosting modes as if they were one quantity. Evidence
journals for both free-tier failures are preserved under
results/day3-v1-four-providers/.
