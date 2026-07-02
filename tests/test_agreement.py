"""Cohen's kappa (§5.8 calibration), hand-computed fixtures."""

import pytest

from memarena.metrics.agreement import cohen_kappa

T, F = True, False


class TestCohenKappa:
    def test_perfect_agreement_is_one(self):
        assert cohen_kappa([T, F, T, F], [T, F, T, F]) == 1.0

    def test_hand_computed_kappa(self):
        # a: 5T/5F, b: 5T/5F, 6 agreements -> po = 0.6, pe = 0.5,
        # kappa = (0.6 - 0.5) / (1 - 0.5) = 0.2
        a = [T, T, T, T, T, F, F, F, F, F]
        b = [T, T, T, F, F, F, F, F, T, T]
        assert cohen_kappa(a, b) == pytest.approx(0.2)

    def test_systematic_disagreement_is_negative(self):
        a = [T, T, F, F]
        b = [F, F, T, T]
        assert cohen_kappa(a, b) == pytest.approx(-1.0)

    def test_degenerate_all_identical_is_one(self):
        assert cohen_kappa([T, T, T], [T, T, T]) == 1.0

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            cohen_kappa([T], [T, F])

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            cohen_kappa([], [])
