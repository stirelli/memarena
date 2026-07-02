from memarena.metrics.deterministic import (
    ItemMetric,
    aggregate_run,
    compute_item_metric,
    content_matches,
    mean_of_defined,
    normalize_content,
    percentile,
    percentile_of_defined,
    recall_at_k,
    reciprocal_rank,
)


class TestNormalizeContent:
    def test_lowercases_and_collapses_whitespace(self):
        assert normalize_content("  My   Dog's Name  ") == "my dog's name"


class TestContentMatches:
    def test_exact_match_after_normalization(self):
        assert content_matches("My dog's name is Biscuit.", "  my dog's name is biscuit.  ")

    def test_fuzzy_match_above_threshold(self):
        assert content_matches("My dog's name is Biscuit.", "My dog's name is Biscuit")

    def test_no_match_for_unrelated_text(self):
        assert not content_matches("My dog's name is Biscuit.", "I like hiking on weekends.")


class TestRecallAtK:
    def test_none_when_no_gold_evidence(self):
        assert recall_at_k(["a", "b"], [], k=5) is None

    def test_hit_within_k(self):
        retrieved = ["irrelevant", "My dog's name is Biscuit.", "also irrelevant"]
        assert recall_at_k(retrieved, ["My dog's name is Biscuit."], k=5) == 1.0

    def test_miss_beyond_k(self):
        retrieved = ["irrelevant 1", "irrelevant 2", "My dog's name is Biscuit."]
        assert recall_at_k(retrieved, ["My dog's name is Biscuit."], k=2) == 0.0

    def test_miss_when_absent(self):
        retrieved = ["irrelevant 1", "irrelevant 2"]
        assert recall_at_k(retrieved, ["My dog's name is Biscuit."], k=5) == 0.0


class TestReciprocalRank:
    def test_none_when_no_gold_evidence(self):
        assert reciprocal_rank(["a"], []) is None

    def test_first_position_hit(self):
        assert reciprocal_rank(["My dog's name is Biscuit.", "other"], ["My dog's name is Biscuit."]) == 1.0

    def test_third_position_hit(self):
        retrieved = ["a", "b", "My dog's name is Biscuit."]
        assert reciprocal_rank(retrieved, ["My dog's name is Biscuit."]) == 1 / 3

    def test_zero_when_not_retrieved(self):
        assert reciprocal_rank(["a", "b"], ["My dog's name is Biscuit."]) == 0.0


class TestMeanOfDefined:
    def test_ignores_none(self):
        assert mean_of_defined([1.0, None, 0.0, None]) == 0.5

    def test_empty_or_all_none_is_none(self):
        assert mean_of_defined([]) is None
        assert mean_of_defined([None, None]) is None


class TestPercentile:
    def test_basic(self):
        assert percentile([10.0, 20.0, 30.0], 50) == 20.0

    def test_empty_is_zero(self):
        assert percentile([], 95) == 0.0


class TestPercentileOfDefined:
    def test_ignores_none_values(self):
        assert percentile_of_defined([None, 10.0, 20.0, None], 50) == 15.0

    def test_all_none_returns_none(self):
        assert percentile_of_defined([None, None], 50) is None


class TestComputeItemMetric:
    def test_computes_recall_at_multiple_k_and_rr(self):
        retrieved = ["a", "b", "My dog's name is Biscuit.", "d"]
        metric = compute_item_metric(
            item_id="smoke-001",
            retrieved_contents=retrieved,
            gold_evidence=["My dog's name is Biscuit."],
            add_latency_ms=12.0,
            search_latency_ms=34.0,
            k_values=(1, 3, 5, 10),
        )
        assert isinstance(metric, ItemMetric)
        assert metric.verbatim_recall_at_k[1] == 0.0
        assert metric.verbatim_recall_at_k[3] == 1.0
        assert metric.verbatim_recall_at_k[5] == 1.0
        assert metric.verbatim_reciprocal_rank == 1 / 3
        assert metric.add_latency_ms == 12.0
        assert metric.search_latency_ms == 34.0

    def test_abstention_item_has_none_recall_and_rr(self):
        metric = compute_item_metric(
            item_id="smoke-017", retrieved_contents=["a", "b"], gold_evidence=[],
            add_latency_ms=5.0, search_latency_ms=5.0, k_values=(1, 5),
        )
        assert metric.verbatim_recall_at_k == {1: None, 5: None}
        assert metric.verbatim_reciprocal_rank is None


class TestAggregateRun:
    def test_aggregates_recall_mrr_and_latency_percentiles(self):
        items = [
            compute_item_metric("i1", ["gold text"], ["gold text"], 10.0, 20.0, k_values=(1, 5)),
            compute_item_metric("i2", ["other"], ["gold text"], 30.0, 40.0, k_values=(1, 5)),
            compute_item_metric("i3", ["a"], [], 50.0, 60.0, k_values=(1, 5)),  # abstention, excluded from recall/MRR
        ]
        run = aggregate_run(items, k_values=(1, 5))

        assert run.n_items == 3
        assert run.n_scored_items == 2
        assert run.verbatim_recall_at_k[1] == 0.5   # (1.0 + 0.0) / 2, abstention item excluded
        assert run.verbatim_mrr == 0.5              # (1.0 + 0.0) / 2
        assert run.add_latency_p50_ms == percentile([10.0, 30.0, 50.0], 50)
        assert run.search_latency_p95_ms == percentile([20.0, 40.0, 60.0], 95)

    def test_recall_and_mrr_are_none_when_no_item_has_gold_evidence(self):
        items = [
            compute_item_metric("i1", ["x"], [], add_latency_ms=5.0, search_latency_ms=10.0),
            compute_item_metric("i2", ["y"], [], add_latency_ms=None, search_latency_ms=12.0),
        ]
        metrics = aggregate_run(items)
        assert metrics.verbatim_recall_at_k[5] is None
        assert metrics.verbatim_mrr is None
        assert metrics.n_scored_items == 0

    def test_add_latency_percentiles_ignore_reused_ingestion_items(self):
        items = [
            compute_item_metric("i1", ["x"], ["x"], add_latency_ms=100.0, search_latency_ms=1.0),
            compute_item_metric("i2", ["x"], ["x"], add_latency_ms=None, search_latency_ms=1.0),
            compute_item_metric("i3", ["x"], ["x"], add_latency_ms=None, search_latency_ms=1.0),
        ]
        metrics = aggregate_run(items)
        assert metrics.add_latency_p50_ms == 100.0  # only the one real ingest counts
