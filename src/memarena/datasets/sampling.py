from __future__ import annotations

import random

from memarena.datasets.base import QAItem


def stratified_sample(items: list[QAItem], *, sample: int, seed: int,
                      stratify_by: str | None) -> list[QAItem]:
    """Seeded, PROPORTIONAL stratified sample (§5.6).

    Allocation is largest-remainder (Hamilton) over stratum sizes:
    quota_i = floor(sample * n_i / N), leftover seats go to the largest
    fractional remainders (ties broken by stratum name for determinism).
    Because sample < N implies floor(sample * n_i / N) + 1 <= n_i, a quota
    never exceeds its stratum, so the result always has exactly `sample`
    items — no silent shortfall.
    """
    if sample >= len(items):
        return sorted(items, key=lambda i: i.id)
    rng = random.Random(seed)
    if not stratify_by:
        return sorted(rng.sample(items, sample), key=lambda i: i.id)

    strata: dict[str, list[QAItem]] = {}
    for item in items:
        strata.setdefault(getattr(item, stratify_by), []).append(item)

    total = len(items)
    quotas: dict[str, int] = {}
    remainders: list[tuple[float, str]] = []
    for name, bucket in sorted(strata.items()):
        exact = sample * len(bucket) / total
        quotas[name] = int(exact)
        remainders.append((-(exact - quotas[name]), name))
    for _, name in sorted(remainders)[: sample - sum(quotas.values())]:
        quotas[name] += 1

    selected: list[QAItem] = []
    for name, bucket in sorted(strata.items()):
        selected.extend(rng.sample(bucket, quotas[name]))
    return sorted(selected, key=lambda i: i.id)
