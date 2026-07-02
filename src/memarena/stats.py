from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Statistical reporting (§5.0.6, §5.10): N repetitions, bootstrap 95% CIs,
# paired per-item deltas between providers on identical items. Differences
# whose 95% CI crosses zero are flagged not significant and rendered with
# an approx marker on the leaderboard, never as a ranking jump.

DEFAULT_N_BOOT = 10_000


@dataclass(frozen=True)
class CIEstimate:
    mean: float
    ci95_low: float
    ci95_high: float
    n: int


@dataclass(frozen=True)
class PairedDelta:
    mean_delta: float  # mean(a - b) over items present in BOTH runs
    ci95_low: float
    ci95_high: float
    n_pairs: int
    significant: bool  # False when the 95% CI crosses zero


def bootstrap_ci(values: list[float], *, seed: int = 42, n_boot: int = DEFAULT_N_BOOT) -> CIEstimate:
    """Seeded percentile bootstrap of the mean. Deterministic given
    (values order-insensitively, seed) — published CIs must reproduce."""
    if not values:
        raise ValueError("bootstrap_ci needs at least one value")
    data = np.asarray(sorted(values), dtype=float)
    rng = np.random.default_rng(seed)
    samples = rng.choice(data, size=(n_boot, len(data)), replace=True).mean(axis=1)
    low, high = np.percentile(samples, [2.5, 97.5])
    return CIEstimate(mean=float(data.mean()), ci95_low=float(low), ci95_high=float(high), n=len(data))


def paired_delta(a_by_item: dict[str, float], b_by_item: dict[str, float],
                 *, seed: int = 42, n_boot: int = DEFAULT_N_BOOT) -> PairedDelta:
    """Paired per-item delta a - b over the intersection of item ids —
    pairing removes item-difficulty variance, which is most of the variance
    on stratified QA samples. Items scored by only one provider (e.g. an
    infra_error on the other side) are excluded, never imputed."""
    shared = sorted(set(a_by_item) & set(b_by_item))
    if not shared:
        raise ValueError("paired_delta needs at least one shared item id")
    deltas = [a_by_item[item] - b_by_item[item] for item in shared]
    est = bootstrap_ci(deltas, seed=seed, n_boot=n_boot)
    significant = not (est.ci95_low <= 0.0 <= est.ci95_high)
    return PairedDelta(mean_delta=est.mean, ci95_low=est.ci95_low, ci95_high=est.ci95_high,
                       n_pairs=len(shared), significant=significant)
