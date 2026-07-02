"""Aggregate Day 3 journals into per-journal summary lines (feeds
results/day3-v1-four-providers/RESULTS.md). Pure journal math — no
network, no provider code.

Run: .venv/bin/python scripts/day3_results_summary.py [results_dir]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def _mean(values):
    values = [v for v in values if v is not None]
    return round(float(sum(values) / len(values)), 4) if values else None


def _pct(values, p):
    values = [v for v in values if v is not None]
    return round(float(np.percentile(values, p)), 1) if values else None


def summarize(path: Path) -> dict:
    ok_rows, errors = [], 0
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if row["status"] == "ok":
            ok_rows.append(row)
        else:
            errors += 1
    return {
        "journal": path.name,
        "ok": len(ok_rows),
        "infra_errors": errors,
        # journals written before the verbatim rename keep the old keys
        "verbatim_recall_at_5": _mean([
            (r.get("verbatim_recall_at_k") or r.get("recall_at_k") or {}).get("5") for r in ok_rows]),
        "verbatim_ndcg_at_5": _mean([
            (r.get("verbatim_ndcg_at_k") or r.get("ndcg_at_k") or {}).get("5") for r in ok_rows]),
        "verbatim_mrr": _mean([
            r.get("verbatim_reciprocal_rank", r.get("reciprocal_rank")) for r in ok_rows]),
        "add_p50_ms": _pct([r.get("add_latency_ms") for r in ok_rows], 50),
        "add_p95_ms": _pct([r.get("add_latency_ms") for r in ok_rows], 95),
        "settle_p50_ms": _pct([r.get("settle_latency_ms") for r in ok_rows], 50),
        "search_p50_ms": _pct([r.get("search_latency_ms") for r in ok_rows], 50),
        "search_p95_ms": _pct([r.get("search_latency_ms") for r in ok_rows], 95),
        "metered_cost_usd": round(sum(r.get("cost_usd") or 0.0 for r in ok_rows), 4),
    }


def main() -> None:
    results_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/day3-v1-four-providers")
    for path in sorted(results_dir.glob("*__journal.jsonl")):
        print(json.dumps(summarize(path)))


if __name__ == "__main__":
    main()
