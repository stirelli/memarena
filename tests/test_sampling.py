"""Stratified sampler correctness (adversarial review, 2026-07-02).

The sampler must be (a) seeded-deterministic, (b) PROPORTIONAL per stratum
(largest-remainder allocation), and (c) never silently return fewer items
than requested when the corpus can satisfy the request.

Quota expectations below are hand-computed from the largest-remainder
method: quota_i = floor(sample * n_i / N), then the leftover seats go to
the strata with the largest fractional remainders (ties broken by stratum
name, ascending, for determinism).
"""

from collections import Counter

from memarena.datasets.base import QAItem
from memarena.datasets.sampling import stratified_sample


def _item(id_: str, question_type: str) -> QAItem:
    return QAItem(
        id=id_, namespace="ns", sessions=[], question="q?",
        gold_evidence=[], question_type=question_type,
    )


def _corpus(sizes: dict[str, int]) -> list[QAItem]:
    return [
        _item(f"{stratum}-{i:03d}", stratum)
        for stratum, n in sorted(sizes.items())
        for i in range(n)
    ]


def _stratum_counts(items: list[QAItem]) -> Counter:
    return Counter(i.question_type for i in items)


class TestProportionalAllocation:
    def test_hand_computed_quotas_6_3_1_sample_5(self):
        # N=10, sample=5. Exact: a -> 3.0 (floor 3, rem 0), b -> 1.5
        # (floor 1, rem .5), c -> 0.5 (floor 0, rem .5). One seat left,
        # remainder tie between b and c -> b wins alphabetically.
        items = _corpus({"a": 6, "b": 3, "c": 1})
        picked = stratified_sample(items, sample=5, seed=42, stratify_by="question_type")
        assert len(picked) == 5
        assert _stratum_counts(picked) == Counter({"a": 3, "b": 2})

    def test_hand_computed_quotas_13_6_1_sample_10(self):
        # N=20, sample=10. a -> 6.5 (floor 6, rem .5), b -> 3.0 (floor 3,
        # rem 0), c -> 0.5 (floor 0, rem .5). One seat left, tie a/c -> a.
        items = _corpus({"a": 13, "b": 6, "c": 1})
        picked = stratified_sample(items, sample=10, seed=42, stratify_by="question_type")
        assert len(picked) == 10
        assert _stratum_counts(picked) == Counter({"a": 7, "b": 3})

    def test_skewed_strata_are_not_equally_weighted(self):
        # Regression for the equal-quota bug: with strata 90/10 and
        # sample=10, proportional allocation gives 9/1, never 5/5.
        items = _corpus({"big": 90, "tiny": 10})
        picked = stratified_sample(items, sample=10, seed=42, stratify_by="question_type")
        assert _stratum_counts(picked) == Counter({"big": 9, "tiny": 1})


class TestNoSilentShortfall:
    def test_returns_exactly_the_requested_sample_size(self):
        # Regression: the old equal-quota sampler with strata {a:1, b:9}
        # and sample=6 allocated 3+3, capped a at 1, and silently
        # returned 4 items. Proportional allocation gives a=1, b=5.
        items = _corpus({"a": 1, "b": 9})
        picked = stratified_sample(items, sample=6, seed=42, stratify_by="question_type")
        assert len(picked) == 6
        assert _stratum_counts(picked) == Counter({"a": 1, "b": 5})

    def test_quota_never_exceeds_stratum_size(self):
        items = _corpus({"a": 2, "b": 5, "c": 11})
        for sample in range(1, 18):
            picked = stratified_sample(items, sample=sample, seed=7, stratify_by="question_type")
            assert len(picked) == sample, f"sample={sample}"
            counts = _stratum_counts(picked)
            assert counts["a"] <= 2 and counts["b"] <= 5 and counts["c"] <= 11


class TestSeededDeterminism:
    def test_same_seed_same_selection(self):
        items = _corpus({"a": 8, "b": 5, "c": 3})
        a = stratified_sample(items, sample=7, seed=123, stratify_by="question_type")
        b = stratified_sample(items, sample=7, seed=123, stratify_by="question_type")
        assert [i.id for i in a] == [i.id for i in b]

    def test_output_is_sorted_by_id(self):
        items = _corpus({"a": 8, "b": 5})
        picked = stratified_sample(items, sample=6, seed=9, stratify_by="question_type")
        assert [i.id for i in picked] == sorted(i.id for i in picked)

    def test_unstratified_sample_is_seeded_and_sized(self):
        items = _corpus({"a": 10})
        a = stratified_sample(items, sample=4, seed=5, stratify_by=None)
        b = stratified_sample(items, sample=4, seed=5, stratify_by=None)
        assert len(a) == 4
        assert [i.id for i in a] == [i.id for i in b]

    def test_sample_at_least_corpus_size_returns_everything(self):
        items = _corpus({"a": 3, "b": 2})
        picked = stratified_sample(items, sample=5, seed=1, stratify_by="question_type")
        assert len(picked) == 5
        picked = stratified_sample(items, sample=50, seed=1, stratify_by="question_type")
        assert len(picked) == 5
