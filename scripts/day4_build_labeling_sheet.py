"""Day 4: build the hand-labeling sheets from the generated answers.

Two CSVs under calibration/, graded against calibration/rubric.v1.md:

- labeling_sheet_answer_correctness.csv: ~160 (item x provider) rows,
  stratified by question type within each available provider. Doubles as
  the abstention sheet (label_abstained column on the same rows).
- labeling_sheet_evidence_coverage.csv: ~60 rows, half from extractive
  rows and half from abstractive rows (docs/METHODOLOGY_NOTES.md), so
  Cohen's kappa is measured on both representations.

Selection is seeded and deterministic given the answer journals.

Run: .venv/bin/python scripts/day4_build_labeling_sheet.py
"""
from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from pathlib import Path

ANSWERS_DIR = Path("results/day4-answers")
OUT_DIR = Path("calibration")
SEED = 42
ANSWER_SHEET_TARGET = 160
COVERAGE_SHEET_TARGET = 60


def _load_rows() -> list[dict]:
    rows = []
    for path in sorted(ANSWERS_DIR.glob("*__answers.jsonl")):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row.get("status") == "ok":
                rows.append(row)
    if not rows:
        raise SystemExit(f"no answer journals under {ANSWERS_DIR}; run day4_generate_answers.py first")
    return rows


def _stratified_pick(rows: list[dict], target: int, rng: random.Random) -> list[dict]:
    """Proportional largest-remainder over question_type, mirroring the
    dataset sampler's allocation discipline."""
    strata: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        strata[row["question_type"]].append(row)
    total = len(rows)
    target = min(target, total)
    quotas, remainders = {}, []
    for name, bucket in sorted(strata.items()):
        exact = target * len(bucket) / total
        quotas[name] = int(exact)
        remainders.append((-(exact - int(exact)), name))
    for _, name in sorted(remainders)[: target - sum(quotas.values())]:
        quotas[name] += 1
    picked = []
    for name, bucket in sorted(strata.items()):
        picked.extend(rng.sample(bucket, min(quotas[name], len(bucket))))
    return picked


def build_answer_sheet(rows: list[dict], rng: random.Random) -> None:
    providers = sorted({r["provider"] for r in rows})
    per_provider = ANSWER_SHEET_TARGET // len(providers)
    picked = []
    for provider in providers:
        picked.extend(_stratified_pick([r for r in rows if r["provider"] == provider], per_provider, rng))
    rng.shuffle(picked)  # graders must not label provider-blocked

    out = OUT_DIR / "labeling_sheet_answer_correctness.csv"
    with out.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sheet_id", "provider", "item_id", "question_type", "answerable",
                         "question", "gold_answer", "answer_text",
                         "label_correct", "label_abstained", "notes"])
        for i, row in enumerate(picked, 1):
            writer.writerow([f"ac-{i:03d}", row["provider"], row["item_id"], row["question_type"],
                             row["answerable"], row["question"], row["gold_answer"],
                             row["answer_text"], "", "", ""])
    print(f"answer-correctness sheet: {len(picked)} rows -> {out}", flush=True)


def build_coverage_sheet(rows: list[dict], rng: random.Random) -> None:
    answerable = [r for r in rows if r["answerable"] and r["gold_evidence"]]
    by_repr: dict[str, list[dict]] = defaultdict(list)
    for row in answerable:
        by_repr[row["memory_representation"]].append(row)
    half = COVERAGE_SHEET_TARGET // 2
    picked = []
    for representation in ("extractive", "abstractive"):
        pool = by_repr.get(representation, [])
        if not pool:
            print(f"WARNING: no {representation} rows available for the coverage sheet", flush=True)
            continue
        providers = sorted({r["provider"] for r in pool})
        per_provider = max(1, half // len(providers))
        for provider in providers:
            picked.extend(_stratified_pick(
                [r for r in pool if r["provider"] == provider], per_provider, rng))
    rng.shuffle(picked)

    out = OUT_DIR / "labeling_sheet_evidence_coverage.csv"
    with out.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sheet_id", "provider", "memory_representation", "item_id", "question_type",
                         "question", "gold_answer", "gold_evidence", "retrieved_memories",
                         "label_covered", "notes"])
        for i, row in enumerate(picked, 1):
            evidence = "\n---\n".join(row["gold_evidence"])
            retrieved = "\n---\n".join(r["content"] for r in row["retrieved"])
            writer.writerow([f"ec-{i:03d}", row["provider"], row["memory_representation"],
                             row["item_id"], row["question_type"], row["question"],
                             row["gold_answer"], evidence, retrieved, "", ""])
    print(f"evidence-coverage sheet: {len(picked)} rows -> {out}", flush=True)


def main() -> None:
    rows = _load_rows()
    rng = random.Random(SEED)
    OUT_DIR.mkdir(exist_ok=True)
    build_answer_sheet(rows, rng)
    build_coverage_sheet(rows, rng)


if __name__ == "__main__":
    main()
