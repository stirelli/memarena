"""Day 4: Cohen's kappa, judge vs human, per judged metric (§5.8).

Reads the hand-labeled sheets in calibration/ (label_* columns filled by
the human grader) and the verdicts journals in results/day4-judgments/,
joins on (provider, item_id), and reports kappa plus the confusion cases.
Ship gate: kappa >= 0.75 per metric; below the gate, iterate the judge
prompt (which bumps its version) and re-run.

Writes calibration/kappa_report.json and prints a summary.

Run: .venv/bin/python scripts/day4_compute_kappa.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from memarena.metrics.agreement import cohen_kappa

JUDGMENTS_DIR = Path("results/day4-judgments")
OUT_PATH = Path("calibration/kappa_report.json")
GATE = 0.75

METRICS = (
    # (metric, sheet path, sheet label column, verdict key)
    ("answer_correctness", Path("calibration/labeling_sheet_answer_correctness.csv"),
     "label_correct", "correct"),
    ("abstention", Path("calibration/labeling_sheet_answer_correctness.csv"),
     "label_abstained", "abstained"),
    ("evidence_coverage", Path("calibration/labeling_sheet_evidence_coverage.csv"),
     "label_covered", "covered"),
)

TRUTHY = {"true", "t", "yes", "y", "1", "correct", "covered", "abstained"}
FALSY = {"false", "f", "no", "n", "0", "incorrect", "not covered", "not abstained"}


def _parse_label(raw: str) -> bool | None:
    value = raw.strip().lower()
    if value in TRUTHY:
        return True
    if value in FALSY:
        return False
    return None  # unlabeled or unparseable — excluded, counted in the report


def _verdicts() -> dict[tuple[str, str], dict]:
    verdicts = {}
    for path in JUDGMENTS_DIR.glob("*__verdicts.jsonl"):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            verdicts[(row["provider"], row["item_id"])] = row["verdicts"]
    if not verdicts:
        raise SystemExit(f"no verdicts under {JUDGMENTS_DIR}; run day4_run_judges.py first")
    return verdicts


def main() -> None:
    verdicts = _verdicts()
    report: dict = {"gate": GATE, "metrics": {}}
    for metric, sheet_path, label_column, verdict_key in METRICS:
        if not sheet_path.exists():
            print(f"[{metric}] sheet missing: {sheet_path}", flush=True)
            continue
        human, judge, disagreements, unlabeled = [], [], [], 0
        with sheet_path.open() as f:
            for row in csv.DictReader(f):
                label = _parse_label(row.get(label_column, ""))
                if label is None:
                    unlabeled += 1
                    continue
                verdict = verdicts.get((row["provider"], row["item_id"]), {}).get(metric)
                if verdict is None:
                    continue
                human.append(label)
                judge.append(bool(verdict[verdict_key]))
                if human[-1] != judge[-1]:
                    disagreements.append({
                        "sheet_id": row["sheet_id"], "provider": row["provider"],
                        "item_id": row["item_id"], "human": human[-1], "judge": judge[-1],
                        "judge_reasoning": verdict.get("reasoning", ""),
                    })
        if not human:
            print(f"[{metric}] no labeled+judged pairs yet ({unlabeled} unlabeled rows)", flush=True)
            continue
        kappa = cohen_kappa(human, judge)
        report["metrics"][metric] = {
            "kappa": round(kappa, 4), "n": len(human), "unlabeled_rows": unlabeled,
            "gate_passed": kappa >= GATE, "disagreements": disagreements,
        }
        print(f"[{metric}] kappa={kappa:.3f} (n={len(human)}, gate {'PASS' if kappa >= GATE else 'FAIL'}, "
              f"{len(disagreements)} disagreements)", flush=True)

    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=2))
    print(f"report -> {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
