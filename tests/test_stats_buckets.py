"""Day 4: stats (bootstrap CIs, paired deltas) and the representation-aware
failure-bucket cascade. Hand-checked expectations where values are exact."""

import pytest

from memarena.buckets import BucketAssignment, assign_bucket, ops_outlier_threshold_ms
from memarena.stats import bootstrap_ci, paired_delta


class TestBootstrapCI:
    def test_constant_values_have_degenerate_ci(self):
        est = bootstrap_ci([0.5] * 20)
        assert est.mean == 0.5
        assert est.ci95_low == 0.5
        assert est.ci95_high == 0.5
        assert est.n == 20

    def test_deterministic_given_seed_and_order_insensitive(self):
        a = bootstrap_ci([0.0, 1.0, 1.0, 0.0, 1.0], seed=7)
        b = bootstrap_ci([1.0, 1.0, 0.0, 1.0, 0.0], seed=7)
        assert a == b

    def test_ci_brackets_the_mean(self):
        est = bootstrap_ci([0.0] * 50 + [1.0] * 50)
        assert est.ci95_low <= est.mean <= est.ci95_high
        assert est.ci95_low > 0.3 and est.ci95_high < 0.7

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            bootstrap_ci([])


class TestPairedDelta:
    def test_pairs_only_shared_items(self):
        a = {"i1": 1.0, "i2": 1.0, "only_a": 1.0}
        b = {"i1": 0.0, "i2": 1.0, "only_b": 0.0}
        delta = paired_delta(a, b)
        assert delta.n_pairs == 2
        assert delta.mean_delta == pytest.approx(0.5)

    def test_clear_difference_is_significant(self):
        a = {f"i{n}": 1.0 for n in range(40)}
        b = {f"i{n}": 0.0 for n in range(40)}
        assert paired_delta(a, b).significant is True

    def test_zero_delta_is_not_significant(self):
        a = {f"i{n}": float(n % 2) for n in range(40)}
        delta = paired_delta(a, dict(a))
        assert delta.mean_delta == 0.0
        assert delta.significant is False

    def test_disjoint_items_raise(self):
        with pytest.raises(ValueError):
            paired_delta({"a": 1.0}, {"b": 1.0})


class TestBucketCascade:
    def _assign(self, **overrides):
        defaults = dict(answerable=True, abstained=False, correct=False, retrieval_ok=True,
                        retrieval_detector="verbatim", question_type="multi-session")
        return assign_bucket(**{**defaults, **overrides})

    def test_answered_unanswerable_is_abstention_fail(self):
        got = self._assign(answerable=False, abstained=False, correct=False)
        assert got == BucketAssignment("abstention_fail", "answered_unanswerable")

    def test_abstained_answerable_is_abstention_fail_even_if_retrieval_missed(self):
        got = self._assign(abstained=True, retrieval_ok=False)
        assert got.bucket == "abstention_fail"  # cascade order: abstention first

    def test_correct_abstention_passes(self):
        got = self._assign(answerable=False, abstained=True)
        assert got.bucket is None

    def test_retrieval_miss_records_the_detector(self):
        verbatim = self._assign(retrieval_ok=False)
        coverage = self._assign(retrieval_ok=False, retrieval_detector="evidence_coverage")
        assert verbatim == BucketAssignment("retrieval_miss", "verbatim")
        assert coverage == BucketAssignment("retrieval_miss", "evidence_coverage")

    def test_ku_failure_after_retrieval_is_update_conflict(self):
        got = self._assign(question_type="knowledge-update")
        assert got.bucket == "update_conflict"

    def test_failure_after_retrieval_is_synthesis_fail(self):
        assert self._assign().bucket == "synthesis_fail"

    def test_missing_retrieval_verdict_never_blames_reader_or_retrieval(self):
        got = self._assign(retrieval_ok=None)
        assert got == BucketAssignment("unattributed_fail", "no_retrieval_verdict")

    def test_correct_item_passes_unless_ops_outlier(self):
        assert self._assign(correct=True).bucket is None
        assert self._assign(correct=True, ops_outlier=True).bucket == "ops_outlier"


class TestOpsOutlierThreshold:
    def test_hand_computed_3x_iqr(self):
        # [10, 20, 30, 40]: Q1 = 17.5, Q3 = 32.5, IQR = 15 -> 32.5 + 45 = 77.5
        assert ops_outlier_threshold_ms([10, 20, 30, 40]) == pytest.approx(77.5)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            ops_outlier_threshold_ms([])
