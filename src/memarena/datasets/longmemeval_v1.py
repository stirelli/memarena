from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Protocol

from memarena.datasets.base import DatasetLoader, QAItem, Session
from memarena.datasets.sampling import stratified_sample

# --- Real schema, confirmed 2026-07-02 against the live origin (see
# datasets/LICENSES.md before re-deriving):
#   longmemeval_s_cleaned.json: list of 500 items, each
#     {question_id, question_type, question, answer (str|int), question_date,
#      haystack_dates, haystack_session_ids, haystack_sessions,
#      answer_session_ids}
#   haystack_sessions: list of sessions aligned index-by-index with
#   haystack_session_ids and haystack_dates; a session is a list of turns
#   {role: user|assistant, content: str} where turns inside evidence
#   sessions additionally carry has_answer: true on the answer-bearing
#   turns. Dates are "YYYY/MM/DD (Dow) HH:MM".
#
# Evidence labels (the reason V1 is the PRIMARY dataset, §8 Day 3): every
# non-abstention item names its evidence sessions (answer_session_ids) and
# flags the answer-bearing turns (has_answer). gold_evidence = the contents
# of those turns, so Recall@k / NDCG@k / MRR are REAL for this dataset —
# unlike LongMemEval-V2 (latency/cost-only secondary track, no evidence
# labels). The 30 abstention items (question_id ends with "_abs") form the
# 7th stratum, are answerable=False, and carry gold_evidence=[] — retrieval
# metrics exclude them (None), abstention grading is Day-4 judge work.
#
# Scope cap (documented, not silent truncation — the published-batch
# protocol is the FULL ~48-session haystack; a capped haystack makes
# retrieval strictly easier and is what fits free-tier provider quotas,
# recorded in datasets/LICENSES.md): each item ingests all of its evidence
# sessions plus earliest distractor sessions up to SESSIONS_PER_ITEM,
# preserving the haystack's chronological order. has_answer flags are
# STRIPPED from the messages handed to providers — labels never leak into
# what a memory system sees.
SESSIONS_PER_ITEM = 8

REVISION = "98d7416c24c778c2fee6e6f3006e7a073259d48f"
ORIGIN_URL = "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned"
ARTIFACT = "longmemeval_s_cleaned.json"
ARTIFACT_SHA256 = "d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442"

ABSTENTION_STRATUM = "abstention"
_DATE_RE = re.compile(r"^(\d{4})/(\d{2})/(\d{2}) \(\w+\) (\d{2}):(\d{2})$")


class ArtifactFetcher(Protocol):
    """Injectable network boundary — the real implementation is
    `_hf_fetch.PinnedFileFetcher` (pinned revision URL, hard sha256
    verification including re-hash of the local cache); tests supply a
    fixture-backed fake so loader unit tests never touch the network."""

    def fetch(self) -> str: ...


def _to_iso(date: str) -> str:
    """'2023/05/30 (Tue) 23:40' -> '2023-05-30T23:40:00Z'. The format was
    verified against all 500 items' haystack_dates (0 mismatches); parsing
    avoids %a so the result never depends on the process locale."""
    match = _DATE_RE.match(date)
    if not match:
        raise ValueError(f"unexpected LongMemEval date format: {date!r}")
    year, month, day, hour, minute = match.groups()
    return f"{year}-{month}-{day}T{hour}:{minute}:00Z"


def _stratum(raw: dict) -> str:
    return ABSTENTION_STRATUM if str(raw["question_id"]).endswith("_abs") else raw["question_type"]


def _select_session_indices(raw: dict) -> list[int]:
    """All evidence-session indices plus earliest distractors, capped at
    SESSIONS_PER_ITEM, in original (chronological) haystack order."""
    evidence_ids = set(raw["answer_session_ids"])
    indices = [i for i, sid in enumerate(raw["haystack_session_ids"]) if sid in evidence_ids]
    for i, sid in enumerate(raw["haystack_session_ids"]):
        if len(indices) >= SESSIONS_PER_ITEM:
            break
        if sid not in evidence_ids:
            indices.append(i)
    return sorted(indices)


class LongMemEvalV1Loader(DatasetLoader):
    """LongMemEval (V1, `longmemeval_s_cleaned`) — the PRIMARY dataset
    (§8 Day 3, revised). See the module docstring for schema, evidence
    mapping and the scope cap; see datasets/LICENSES.md for licensing.

    Namespace = f"lme_v1_{question_id}" — each question has its own
    haystack in the source data, so items never share ingested state
    (contrast with V2's per-domain shared haystacks)."""

    name = "longmemeval_v1"
    origin_url = ORIGIN_URL
    revision = REVISION
    license = "MIT"
    redistributable = False

    def __init__(self, *, fetcher: ArtifactFetcher | None = None, cache_dir: Path | str | None = None):
        self._fetcher = fetcher or _default_fetcher(cache_dir)
        self._raw_items: list[dict] | None = None
        self._artifact_sha256: str | None = None

    def _load_raw(self) -> None:
        if self._raw_items is not None:
            return
        artifact_path = Path(self._fetcher.fetch())
        content = artifact_path.read_bytes()
        self._artifact_sha256 = hashlib.sha256(content).hexdigest()
        self._raw_items = json.loads(content)

    def sha256(self) -> str:
        if self._artifact_sha256 is None:
            self._load_raw()
        assert self._artifact_sha256 is not None
        return self._artifact_sha256

    def load(self, *, sample: int | None = None, seed: int = 42,
             stratify_by: str | None = "question_type") -> list[QAItem]:
        self._load_raw()
        items = [self._to_qaitem(raw) for raw in self._raw_items]
        # Release the parsed 500-item haystacks (~GBs as Python objects);
        # QAItems keep only their capped sessions. A second load() re-parses.
        self._raw_items = None
        if sample is None:
            return sorted(items, key=lambda i: i.id)
        return stratified_sample(items, sample=sample, seed=seed, stratify_by=stratify_by)

    def _to_qaitem(self, raw: dict) -> QAItem:
        answerable = not str(raw["question_id"]).endswith("_abs")
        sessions = []
        gold_evidence: list[str] = []
        evidence_ids = set(raw["answer_session_ids"])
        for i in _select_session_indices(raw):
            session_id = raw["haystack_session_ids"][i]
            turns = raw["haystack_sessions"][i]
            # has_answer is a label, not data — never handed to providers.
            messages = [{"role": t["role"], "content": t["content"]} for t in turns]
            sessions.append(Session(
                session_id=session_id,
                timestamp=_to_iso(raw["haystack_dates"][i]),
                messages=messages,
            ))
            if answerable and session_id in evidence_ids:
                gold_evidence.extend(t["content"] for t in turns if t.get("has_answer"))
        return QAItem(
            id=raw["question_id"],
            namespace=f"lme_v1_{raw['question_id']}",
            sessions=sessions,
            question=raw["question"],
            gold_evidence=gold_evidence,
            question_type=_stratum(raw),
            answerable=answerable,
            gold_answer=str(raw["answer"]),
        )


def _default_fetcher(cache_dir: Path | str | None) -> ArtifactFetcher:
    from memarena.datasets._hf_fetch import PinnedFileFetcher  # local import: network deps only when needed
    directory = Path(cache_dir) if cache_dir else Path(".cache/longmemeval_v1")
    return PinnedFileFetcher(
        url=f"{ORIGIN_URL}/resolve/{REVISION}/{ARTIFACT}",
        cache_path=directory / ARTIFACT,
        expected_sha256=ARTIFACT_SHA256,
    )
