"""Day 4: apply the versioned judges to generated answers.

Reads results/day4-answers/*__answers.jsonl and writes one verdicts
journal per provider under results/day4-judgments/. Three judged metrics
per row (configs/judges/judge.v1.yaml):

- abstention: every ok row (drives two-sided abstention accuracy);
- answer_correctness: answerable rows;
- evidence_coverage: answerable rows with gold evidence — the
  cross-paradigm retrieval verdict (docs/METHODOLOGY_NOTES.md).

The sqlite verdict cache makes reruns and calibration iterations free:
only rows a changed judge version has never seen hit the API.

--sheet-only judges just the rows referenced by the labeling sheets in
calibration/ (the subset Cohen's kappa is computed on), so calibration
can start before paying for the full grid.

Run: .venv/bin/python scripts/day4_run_judges.py [--sheet-only]
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from dotenv import load_dotenv

from memarena.metrics.judge import JudgeCache, load_judges

ANSWERS_DIR = Path("results/day4-answers")
OUT_DIR = Path("results/day4-judgments")
SHEETS = (
    Path("calibration/labeling_sheet_answer_correctness.csv"),
    Path("calibration/labeling_sheet_evidence_coverage.csv"),
)


def _sheet_keys() -> set[tuple[str, str]]:
    keys = set()
    for sheet in SHEETS:
        if not sheet.exists():
            continue
        with sheet.open() as f:
            for row in csv.DictReader(f):
                keys.add((row["provider"], row["item_id"]))
    if not keys:
        raise SystemExit("no labeling sheets found; run day4_build_labeling_sheet.py first")
    return keys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sheet-only", action="store_true")
    args = parser.parse_args()

    load_dotenv()
    judges = load_judges(cache=JudgeCache())
    only = _sheet_keys() if args.sheet_only else None
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for answers_path in sorted(ANSWERS_DIR.glob("*__answers.jsonl")):
        provider = answers_path.name.split("__")[0]
        out_path = OUT_DIR / f"{provider}__verdicts.jsonl"
        n_done = 0
        with out_path.open("w") as out:
            for line in answers_path.read_text().splitlines():
                row = json.loads(line)
                if row.get("status") != "ok":
                    continue
                if only is not None and (row["provider"], row["item_id"]) not in only:
                    continue
                verdicts: dict = {
                    "abstention": judges["abstention"].judge({"answer": row["answer_text"]}),
                }
                if row["answerable"]:
                    verdicts["answer_correctness"] = judges["answer_correctness"].judge({
                        "question": row["question"],
                        "gold_answer": row["gold_answer"],
                        "answer": row["answer_text"],
                    })
                    if row["gold_evidence"]:
                        verdicts["evidence_coverage"] = judges["evidence_coverage"].judge({
                            "question": row["question"],
                            "gold_answer": row["gold_answer"],
                            "gold_evidence": row["gold_evidence"],
                            "retrieved_memories": [r["content"] for r in row["retrieved"]],
                        })
                out.write(json.dumps({
                    "item_id": row["item_id"], "provider": provider,
                    "question_type": row["question_type"], "answerable": row["answerable"],
                    "memory_representation": row["memory_representation"],
                    "verdicts": verdicts,
                }) + "\n")
                out.flush()
                n_done += 1
                if n_done % 25 == 0:
                    print(f"  [{provider}] {n_done} rows judged", flush=True)
        print(f"[{provider}] {n_done} rows -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
