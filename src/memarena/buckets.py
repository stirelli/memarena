from __future__ import annotations

from dataclasses import dataclass

# Failure-bucket taxonomy (§5.9), deterministic cascade, first match wins.
# Day 4 revision (docs/METHODOLOGY_NOTES.md): the retrieval_miss decision is
# representation-aware — verbatim gold matching for extractive stores,
# judged evidence_coverage for abstractive stores — otherwise every
# abstractive failure would be misclassified as a retrieval miss. Every
# assignment records which detector decided it, so bucket distributions
# stay auditable per provider.

BUCKETS = (
    "abstention_fail",
    "retrieval_miss",
    "update_conflict",
    "synthesis_fail",
    "unattributed_fail",
    "ops_outlier",
)

KNOWLEDGE_UPDATE_TYPE = "knowledge-update"


@dataclass(frozen=True)
class BucketAssignment:
    bucket: str | None  # None = the item passed (correct, no ops anomaly)
    decided_by: str


def assign_bucket(
    *,
    answerable: bool,
    abstained: bool,
    correct: bool,
    retrieval_ok: bool | None,
    retrieval_detector: str,  # "verbatim" (extractive) | "evidence_coverage" (abstractive)
    question_type: str,
    ops_outlier: bool = False,
) -> BucketAssignment:
    """Cascade order per §5.9. `retrieval_ok` is the representation-aware
    verdict (verbatim hit for extractive rows, judged coverage for
    abstractive rows); None means no verdict was available, in which case
    a failed item lands in unattributed_fail rather than silently blaming
    retrieval or the reader.

    Documented approximation (until old-fact labels exist): a
    knowledge-update item that failed AFTER successful retrieval is
    bucketed update_conflict without distinguishing stale-fact answers
    from plain synthesis failures inside that stratum."""
    if not answerable and not abstained:
        return BucketAssignment("abstention_fail", "answered_unanswerable")
    if answerable and abstained:
        return BucketAssignment("abstention_fail", "abstained_answerable")
    if not answerable:
        return BucketAssignment(None, "abstained_correctly")

    if correct:
        if ops_outlier:
            return BucketAssignment("ops_outlier", "latency_iqr")
        return BucketAssignment(None, "passed")

    if retrieval_ok is False:
        return BucketAssignment("retrieval_miss", retrieval_detector)
    if retrieval_ok is None:
        return BucketAssignment("unattributed_fail", "no_retrieval_verdict")
    if question_type == KNOWLEDGE_UPDATE_TYPE:
        return BucketAssignment("update_conflict", "ku_failed_after_retrieval")
    return BucketAssignment("synthesis_fail", retrieval_detector)


def ops_outlier_threshold_ms(latencies_ms: list[float]) -> float:
    """3x IQR above Q3 (§5.9 item 5). Callers precompute per run and pass
    per-item booleans into assign_bucket."""
    import numpy as np
    if not latencies_ms:
        raise ValueError("need latencies to compute an outlier threshold")
    q1, q3 = np.percentile(np.asarray(latencies_ms, dtype=float), [25, 75])
    return float(q3 + 3.0 * (q3 - q1))
