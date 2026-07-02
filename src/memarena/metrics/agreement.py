from __future__ import annotations

# Judge calibration statistics (§5.8): Cohen's kappa between the LLM judge
# and the human labels, published per judged metric. Ship gate: >= 0.75.


def cohen_kappa(labels_a: list[bool], labels_b: list[bool]) -> float:
    """Cohen's kappa for two binary raters over the same items.

    kappa = (p_o - p_e) / (1 - p_e); returns 1.0 when both observed and
    chance agreement are perfect (the 0/0 case: all labels identical in
    both raters)."""
    if len(labels_a) != len(labels_b):
        raise ValueError(f"label lists differ in length: {len(labels_a)} vs {len(labels_b)}")
    if not labels_a:
        raise ValueError("cohen_kappa needs at least one pair of labels")
    n = len(labels_a)
    observed = sum(1 for a, b in zip(labels_a, labels_b, strict=True) if a == b) / n
    p_a_true = sum(labels_a) / n
    p_b_true = sum(labels_b) / n
    expected = p_a_true * p_b_true + (1 - p_a_true) * (1 - p_b_true)
    if expected == 1.0:
        return 1.0 if observed == 1.0 else 0.0
    return (observed - expected) / (1 - expected)
