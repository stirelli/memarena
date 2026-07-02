"""Hand-computed metric fixtures (adversarial correctness review, 2026-07-02).

Every expected value in this module was derived by hand from the metric
definitions (spec §5.7), NOT by running the code first. If any assertion here
disagrees with the implementation, the implementation is wrong.

Recall@k: 1.0 if any gold evidence content-matches a top-k retrieved record,
          else 0.0; None (excluded) when the item has no gold evidence.
RR:       1/rank of the first retrieved record matching any gold evidence,
          0.0 if never matched; None when no gold evidence.
Aggregation: arithmetic mean over defined (non-None) values only; latency
          percentiles are numpy linear-interpolation percentiles over
          defined values only.
"""

import math

import pytest

from memarena.metrics.deterministic import (
    aggregate_run,
    compute_item_metric,
    content_matches,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)

GOLD = "the treasure is buried under the old oak tree"
OTHER_A = "completely unrelated sentence about quarterly reports"
OTHER_B = "another filler memory about grocery shopping lists"


class TestFixture1HitAtRankThree:
    """retrieved = [other, other, GOLD, other]; gold = [GOLD].

    Hand computation: first match at rank 3.
      R@1 = 0.0 (top-1 = [other])
      R@3 = 1.0 (GOLD is 3rd)
      R@5 = 1.0
      RR  = 1/3
    """

    retrieved = [OTHER_A, OTHER_B, GOLD, "yet more filler text here"]

    def test_recall_values(self):
        assert recall_at_k(self.retrieved, [GOLD], k=1) == 0.0
        assert recall_at_k(self.retrieved, [GOLD], k=2) == 0.0
        assert recall_at_k(self.retrieved, [GOLD], k=3) == 1.0
        assert recall_at_k(self.retrieved, [GOLD], k=5) == 1.0

    def test_reciprocal_rank(self):
        assert reciprocal_rank(self.retrieved, [GOLD]) == pytest.approx(1 / 3)


class TestFixture2KLargerThanResultCount:
    """retrieved has only 2 records; k = 10 must consider all of them,
    never crash, never fabricate misses.

    Hand computation: match at rank 2.
      R@1  = 0.0
      R@10 = 1.0 (top-10 of a 2-element list is the whole list)
      RR   = 1/2
    Empty retrieval with gold present is a hard miss: R@k = 0.0, RR = 0.0.
    """

    retrieved = [OTHER_A, GOLD]

    def test_k_exceeding_results_uses_what_exists(self):
        assert recall_at_k(self.retrieved, [GOLD], k=10) == 1.0
        assert recall_at_k(self.retrieved, [GOLD], k=1) == 0.0
        assert reciprocal_rank(self.retrieved, [GOLD]) == 0.5

    def test_empty_retrieval_is_a_miss_not_a_crash(self):
        assert recall_at_k([], [GOLD], k=5) == 0.0
        assert reciprocal_rank([], [GOLD]) == 0.0


class TestFixture3AbstentionExclusion:
    """Items with no gold evidence (abstention) are None at item level and
    excluded from run-level means, not counted as 0.0.

    Hand computation for the aggregate below:
      item1: hit at rank 1 -> R@1 = 1.0, RR = 1.0
      item2: abstention    -> None, excluded
      item3: total miss    -> R@1 = 0.0, RR = 0.0
      run R@1 = (1.0 + 0.0) / 2 = 0.5   (NOT (1+0+0)/3 = 0.333...)
      run MRR = (1.0 + 0.0) / 2 = 0.5
      n_items = 3, n_scored_items = 2
    """

    def test_abstention_item_is_none(self):
        assert recall_at_k([OTHER_A], [], k=5) is None
        assert reciprocal_rank([OTHER_A], []) is None

    def test_run_mean_excludes_abstention(self):
        items = [
            compute_item_metric("hit", [GOLD], [GOLD], 1.0, 1.0, k_values=(1,)),
            compute_item_metric("abstain", [OTHER_A], [], 1.0, 1.0, k_values=(1,)),
            compute_item_metric("miss", [OTHER_A], [GOLD], 1.0, 1.0, k_values=(1,)),
        ]
        run = aggregate_run(items, k_values=(1,))
        assert run.recall_at_k[1] == 0.5
        assert run.mrr == 0.5
        assert run.n_items == 3
        assert run.n_scored_items == 2


class TestFixture4DuplicateMemories:
    """Duplicate retrieved records occupy real ranks; they are neither
    collapsed nor deduplicated.

    Hand computation:
      [dup, dup, GOLD]: first match at rank 3 -> RR = 1/3, R@2 = 0.0
      [GOLD, GOLD, other]: first match at rank 1 -> RR = 1.0; the second
        duplicate changes nothing.
      Duplicate GOLD entries in the gold list must not double-count:
        R@k stays binary 1.0.
    """

    def test_duplicate_fillers_push_gold_down(self):
        retrieved = [OTHER_A, OTHER_A, GOLD]
        assert reciprocal_rank(retrieved, [GOLD]) == pytest.approx(1 / 3)
        assert recall_at_k(retrieved, [GOLD], k=2) == 0.0
        assert recall_at_k(retrieved, [GOLD], k=3) == 1.0

    def test_duplicate_gold_retrievals_still_binary(self):
        retrieved = [GOLD, GOLD, OTHER_A]
        assert reciprocal_rank(retrieved, [GOLD]) == 1.0
        assert recall_at_k(retrieved, [GOLD], k=1) == 1.0

    def test_duplicate_gold_evidence_entries_still_binary(self):
        retrieved = [GOLD, OTHER_A]
        assert recall_at_k(retrieved, [GOLD, GOLD], k=5) == 1.0
        assert reciprocal_rank(retrieved, [GOLD, GOLD]) == 1.0


class TestFixture5MultiGoldAndAggregatePercentiles:
    """(a) Multiple gold evidence entries: any of them matching counts
    (rank of the FIRST retrieved record matching ANY gold).

    (b) Gold-evidence matching: exact after normalization, fuzzy only for
    near-identical strings (trailing punctuation), never for unrelated text.

    (c) Full-run aggregate with hand-computed means and numpy linear
    percentiles:
      items (RR):   A = 1/3, B = 1.0, C = None (abstention), D = 0.0
      MRR = (1/3 + 1 + 0) / 3 = 4/9
      R@1 = (0 + 1 + 0) / 3 = 1/3;  R@3 = (1 + 1 + 0) / 3 = 2/3
      add latencies [10, None, 50, 30] -> defined [10, 30, 50]
        p50 = 30.0
        p95: index 0.95*(3-1) = 1.9 -> 30 + 0.9*(50-30) = 48.0
      search latencies [20, 40, 60, 80]
        p50: index 1.5 -> 40 + 0.5*20 = 50.0
        p95: index 2.85 -> 60 + 0.85*20 = 77.0
    """

    def test_second_gold_matching_at_rank_one(self):
        gold = ["never retrieved evidence", GOLD]
        assert reciprocal_rank([GOLD, OTHER_A], gold) == 1.0
        assert recall_at_k([GOLD, OTHER_A], gold, k=1) == 1.0

    def test_fuzzy_matches_trailing_punctuation_only(self):
        assert content_matches("alpha beta gamma delta", "Alpha beta gamma delta.")
        assert not content_matches("alpha beta gamma delta", "epsilon zeta eta theta")

    def test_aggregate_exact_values(self):
        items = [
            compute_item_metric("A", [OTHER_A, OTHER_B, GOLD], [GOLD], 10.0, 20.0, k_values=(1, 3)),
            compute_item_metric("B", [GOLD], [GOLD], None, 40.0, k_values=(1, 3)),
            compute_item_metric("C", [OTHER_A], [], 50.0, 60.0, k_values=(1, 3)),
            compute_item_metric("D", [OTHER_A, OTHER_B], [GOLD], 30.0, 80.0, k_values=(1, 3)),
        ]
        run = aggregate_run(items, k_values=(1, 3))
        assert run.mrr == pytest.approx(4 / 9)
        assert run.recall_at_k[1] == pytest.approx(1 / 3)
        assert run.recall_at_k[3] == pytest.approx(2 / 3)
        assert run.n_items == 4
        assert run.n_scored_items == 3
        assert run.add_latency_p50_ms == pytest.approx(30.0)
        assert run.add_latency_p95_ms == pytest.approx(48.0)
        assert run.search_latency_p50_ms == pytest.approx(50.0)
        assert run.search_latency_p95_ms == pytest.approx(77.0)


class TestFixture6NDCG:
    """NDCG@k, binary relevance, gold-consuming gains (§8 Day 3; the
    LongMemEval paper's official retrieval metric).

    Hand computations (log2 discount, rank r contributes 1/log2(r+1)):
      (a) [other, other, GOLD], gold=[GOLD], k=3:
          DCG = 1/log2(4) = 0.5; IDCG = 1/log2(2) = 1.0 -> NDCG = 0.5
          k=1: DCG = 0 -> NDCG = 0.0
      (b) [GOLD, ...anything], k>=1 -> NDCG = 1.0 with one gold
      (c) two golds, retrieved [G1, other, G2], k=3:
          DCG = 1 + 1/log2(4) = 1.5; IDCG = 1 + 1/log2(3)
      (d) duplicate retrieval of the SAME gold earns nothing twice:
          [GOLD, GOLD], gold=[GOLD], k=2 -> DCG = 1.0 = IDCG -> 1.0, never >1
      (e) duplicate gold entries dedupe for IDCG: gold=[GOLD, GOLD],
          retrieved=[GOLD] -> 1.0
      (f) no gold -> None; empty retrieval with gold -> 0.0
    """

    GOLD_2 = "the second treasure is hidden behind the waterfall"

    def test_hit_at_rank_three(self):
        retrieved = [OTHER_A, OTHER_B, GOLD]
        assert ndcg_at_k(retrieved, [GOLD], k=3) == pytest.approx(0.5)
        assert ndcg_at_k(retrieved, [GOLD], k=1) == 0.0

    def test_hit_at_rank_one_is_perfect(self):
        assert ndcg_at_k([GOLD, OTHER_A], [GOLD], k=5) == pytest.approx(1.0)

    def test_two_golds_split_ranks(self):
        retrieved = [GOLD, OTHER_A, self.GOLD_2]
        expected = 1.5 / (1.0 + 1.0 / math.log2(3))
        assert ndcg_at_k(retrieved, [GOLD, self.GOLD_2], k=3) == pytest.approx(expected)

    def test_duplicate_retrieval_never_exceeds_one(self):
        assert ndcg_at_k([GOLD, GOLD], [GOLD], k=2) == pytest.approx(1.0)

    def test_duplicate_gold_entries_dedupe(self):
        assert ndcg_at_k([GOLD], [GOLD, GOLD], k=2) == pytest.approx(1.0)

    def test_no_gold_is_none_and_empty_retrieval_is_zero(self):
        assert ndcg_at_k([OTHER_A], [], k=5) is None
        assert ndcg_at_k([], [GOLD], k=5) == 0.0

    def test_aggregate_ndcg_excludes_abstention(self):
        items = [
            compute_item_metric("A", [OTHER_A, OTHER_B, GOLD], [GOLD], 10.0, 20.0, k_values=(1, 3)),
            compute_item_metric("B", [GOLD], [GOLD], None, 40.0, k_values=(1, 3)),
            compute_item_metric("C", [OTHER_A], [], 50.0, 60.0, k_values=(1, 3)),
            compute_item_metric("D", [OTHER_A, OTHER_B], [GOLD], 30.0, 80.0, k_values=(1, 3)),
        ]
        run = aggregate_run(items, k_values=(1, 3))
        assert run.ndcg_at_k[3] == pytest.approx((0.5 + 1.0 + 0.0) / 3)
        assert run.ndcg_at_k[1] == pytest.approx(1 / 3)

    def test_empty_run_ndcg_is_none(self):
        run = aggregate_run([])
        assert all(v is None for v in run.ndcg_at_k.values())


class TestFixture7ContainmentMatching:
    """Containment fallback in content_matches (§8 Day 3): a fixed-size
    chunk of an evidence turn, or a whole session embedding the evidence
    turn, is a hit — but tiny fragments (< 40 normalized chars) never are.

    LONG_GOLD is 208 chars; its first 200 chars are exactly what
    baseline_rag's chunker would store, and they fail the 0.85 fuzzy ratio
    against the full turn only barely — containment is what makes the
    match deterministic instead of ratio-dependent."""

    LONG_GOLD = (
        "I graduated with a degree in business administration and I just started "
        "a new role as an operations manager at the shipping company downtown, "
        "which I mentioned when we talked about my career plans last spring."
    )

    def test_chunk_of_evidence_turn_matches(self):
        chunk = self.LONG_GOLD[:100]
        assert content_matches(self.LONG_GOLD, chunk)

    def test_session_embedding_evidence_turn_matches(self):
        session_text = "user: hello there\nassistant: hi!\nuser: " + self.LONG_GOLD + "\nassistant: great!"
        assert content_matches(self.LONG_GOLD, session_text)

    def test_tiny_fragment_does_not_match(self):
        assert not content_matches(self.LONG_GOLD, "business administration")

    def test_unrelated_long_text_does_not_match(self):
        unrelated = (
            "the quarterly report shows revenue growth across all seven regions "
            "with particularly strong performance in the northern district offices"
        )
        assert not content_matches(self.LONG_GOLD, unrelated)


class TestEmptyRunHasNoFabricatedNumbers:
    """A run where every item infra-errored aggregates to N/A everywhere.
    A silent 0.0 for search latency would misreport an empty run as an
    instant one (review finding F5)."""

    def test_aggregate_of_nothing_is_all_none(self):
        run = aggregate_run([])
        assert run.n_items == 0
        assert run.n_scored_items == 0
        assert run.mrr is None
        assert all(v is None for v in run.recall_at_k.values())
        assert run.add_latency_p50_ms is None
        assert run.add_latency_p95_ms is None
        assert run.search_latency_p50_ms is None
        assert run.search_latency_p95_ms is None
