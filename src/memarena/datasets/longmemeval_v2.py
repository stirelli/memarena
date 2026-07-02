from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Protocol

from memarena.datasets.base import DatasetLoader, QAItem, Session
from memarena.datasets.sampling import stratified_sample

# --- Real schema, confirmed 2026-07-01 against the live origin (do not
# re-derive without re-checking — see datasets/LICENSES.md):
#   questions.jsonl: {id, domain[web|enterprise], environment, question_type,
#                     question, image, answer, eval_function}
#   trajectories.jsonl: {id, domain, environment, goal, outcome, start_url,
#                        states:[{state_index, step, url, action, thought,
#                        accessibility_tree, screenshot}]}
#   haystacks/lme_v2_small.json: {question_id: [trajectory_id, ...]} — all
#   questions in one domain share the same 100-id array.
#
# There is no evidence-span field anywhere in this dataset — QAItem.gold_evidence
# is always []; Recall@k/MRR report as N/A for this dataset (see
# metrics/deterministic.py's None-safe aggregation). Grading is via
# `eval_function` against a final boxed answer, scoped to Day 4 (judge work).
#
# Scope caps (documented, not silent truncation — a larger paid run can raise
# these): only the first TRAJECTORIES_PER_DOMAIN ids per domain's shared
# haystack are ingested; each trajectory is capped to STATES_PER_TRAJECTORY
# states and ACCESSIBILITY_TREE_CHARS chars/state.
TRAJECTORIES_PER_DOMAIN = 6
STATES_PER_TRAJECTORY = 15
ACCESSIBILITY_TREE_CHARS = 500

REVISION = "f152293e235517d504809563c833d7190b8c713b"
ORIGIN_URL = "https://huggingface.co/datasets/xiaowu0162/longmemeval-v2"


class Fetcher(Protocol):
    """Injectable network boundary — the real implementation (`_hf_fetch.py`)
    downloads from `ORIGIN_URL` at `REVISION` with sha256 verification
    against the origin's `checksums.sha256`; tests supply a fixture-backed
    fake so the loader's unit tests never touch the network."""

    def fetch_questions(self) -> str: ...
    def fetch_haystack_small(self) -> str: ...
    def fetch_trajectories(self, needed_ids: set[str]) -> str: ...


def _question_type_answerable(question_type: str) -> bool:
    return not question_type.endswith("-abs")


def _synthetic_timestamp(index: int) -> str:
    """The source dataset has no capture timestamps. This assigns a
    synthetic, deterministic sequence — documented as non-authoritative."""
    day = 1 + (index % 27)
    return f"2026-01-{day:02d}T00:00:00Z"


def _trajectory_to_session(trajectory: dict, *, index: int) -> Session:
    messages = [{"role": "user", "content": f"Goal: {trajectory['goal']}"}]
    for state in trajectory["states"][:STATES_PER_TRAJECTORY]:
        tree = (state.get("accessibility_tree") or "")[:ACCESSIBILITY_TREE_CHARS]
        content = f"URL: {state['url']}\nAction: {state.get('action')}\nObservation: {tree}"
        messages.append({"role": "assistant", "content": content})
    return Session(session_id=trajectory["id"], timestamp=_synthetic_timestamp(index), messages=messages)


class LongMemEvalV2Loader(DatasetLoader):
    """Real LongMemEval-V2 (§8 Day 2). See module docstring above and
    datasets/LICENSES.md for the full license/schema/scope-cap record.

    Namespace = f"lme_v2_{domain}" (two total: lme_v2_web, lme_v2_enterprise)
    — every question in a domain shares that domain's ingested trajectories,
    matching the source dataset's own per-domain shared-haystack design and
    letting the runner's ingestion cache (§5.3) ingest each domain once.
    """

    name = "longmemeval_v2"
    origin_url = ORIGIN_URL
    revision = REVISION
    license = "Apache-2.0"
    redistributable = False

    def __init__(self, *, fetcher: Fetcher | None = None, cache_dir: Path | str | None = None):
        self._fetcher = fetcher or _default_fetcher(cache_dir)
        self._questions: list[dict] | None = None
        self._sessions_by_domain: dict[str, list[Session]] | None = None

    def _load_raw(self) -> None:
        if self._questions is not None:
            return
        questions_path = self._fetcher.fetch_questions()
        with open(questions_path) as f:
            self._questions = [json.loads(line) for line in f if line.strip()]

        haystack_path = self._fetcher.fetch_haystack_small()
        with open(haystack_path) as f:
            haystack = json.load(f)

        domain_by_question = {q["id"]: q["domain"] for q in self._questions}
        needed_by_domain: dict[str, list[str]] = {}
        for question_id, trajectory_ids in haystack.items():
            domain = domain_by_question.get(question_id)
            if domain and domain not in needed_by_domain:
                needed_by_domain[domain] = trajectory_ids[:TRAJECTORIES_PER_DOMAIN]

        needed_ids = {tid for ids in needed_by_domain.values() for tid in ids}
        trajectories_path = self._fetcher.fetch_trajectories(needed_ids)
        trajectories_by_id: dict[str, dict] = {}
        with open(trajectories_path) as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                if obj["id"] in needed_ids:
                    trajectories_by_id[obj["id"]] = obj

        self._sessions_by_domain = {}
        for domain, ids in needed_by_domain.items():
            sessions = [
                _trajectory_to_session(trajectories_by_id[tid], index=i)
                for i, tid in enumerate(ids) if tid in trajectories_by_id
            ]
            self._sessions_by_domain[domain] = sessions

    def sha256(self) -> str:
        self._load_raw()
        canonical = json.dumps(self._questions, sort_keys=True).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

    def load(self, *, sample: int | None = None, seed: int = 42,
             stratify_by: str | None = "question_type") -> list[QAItem]:
        self._load_raw()
        items = [self._to_qaitem(q) for q in self._questions if q["domain"] in self._sessions_by_domain]
        if sample is None:
            return sorted(items, key=lambda i: i.id)
        return stratified_sample(items, sample=sample, seed=seed, stratify_by=stratify_by)

    def _to_qaitem(self, q: dict) -> QAItem:
        return QAItem(
            id=q["id"],
            namespace=f"lme_v2_{q['domain']}",
            sessions=self._sessions_by_domain[q["domain"]],
            question=q["question"],
            gold_evidence=[],
            question_type=q["question_type"],
            answerable=_question_type_answerable(q["question_type"]),
            gold_answer=q["answer"],
        )

def _default_fetcher(cache_dir: Path | str | None) -> Fetcher:
    from memarena.datasets._hf_fetch import HuggingFaceFetcher  # local import: network deps only when needed
    return HuggingFaceFetcher(
        revision=REVISION,
        cache_dir=Path(cache_dir) if cache_dir else Path(".cache/longmemeval_v2"),
    )
