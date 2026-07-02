"""LongMemEval V1 loader tests (§8 Day 3). The fake artifact below mirrors
the REAL schema confirmed 2026-07-02 against the live origin (see the
loader's module docstring): per-item haystacks, aligned id/date/session
lists, has_answer flags on evidence turns, _abs suffix for abstention."""

import hashlib
import json
from collections import Counter

import pytest

from memarena.datasets._hf_fetch import ChecksumMismatchError, PinnedFileFetcher
from memarena.datasets.longmemeval_v1 import (
    SESSIONS_PER_ITEM,
    LongMemEvalV1Loader,
    _to_iso,
)


def _session(*contents, answer_indices=()):
    turns = []
    for i, content in enumerate(contents):
        turn = {"role": "user" if i % 2 == 0 else "assistant", "content": content}
        if i in answer_indices:
            turn["has_answer"] = True
        turns.append(turn)
    return turns


def _item(question_id, question_type, session_specs, answer_ids, *, answer="the answer"):
    """session_specs: list of (session_id, date, session_turns)."""
    return {
        "question_id": question_id,
        "question_type": question_type,
        "question": f"question for {question_id}?",
        "answer": answer,
        "question_date": "2023/06/01 (Thu) 10:00",
        "haystack_session_ids": [s[0] for s in session_specs],
        "haystack_dates": [s[1] for s in session_specs],
        "haystack_sessions": [s[2] for s in session_specs],
        "answer_session_ids": answer_ids,
    }


EVIDENCE_TURN = "I adopted a golden retriever named Biscuit from the shelter on Elm Street last weekend."
SECOND_EVIDENCE = "Biscuit's vet appointment went well; she weighs thirty pounds already."
FILLER = "Let's talk about the weather and other unrelated small talk topics today."


def _build_raw():
    items = [
        _item(
            "q_ssu_1", "single-session-user",
            [
                ("s1", "2023/05/20 (Sat) 02:21", _session(FILLER, "sure")),
                ("s2", "2023/05/21 (Sun) 09:00", _session(EVIDENCE_TURN, "lovely!", SECOND_EVIDENCE, "noted",
                                                          answer_indices=(0, 2))),
                ("s3", "2023/05/22 (Mon) 11:30", _session(FILLER, "ok")),
            ],
            ["s2"],
        ),
        _item(
            "q_ms_1", "multi-session",
            [
                ("m1", "2023/05/01 (Mon) 08:00", _session(EVIDENCE_TURN, "nice", answer_indices=(0,))),
                ("m2", "2023/05/02 (Tue) 08:00", _session(FILLER, "ok")),
                ("m3", "2023/05/03 (Wed) 08:00", _session(SECOND_EVIDENCE, "great", answer_indices=(0,))),
            ],
            ["m1", "m3"],
        ),
        _item(
            "q_ku_1_abs", "knowledge-update",
            [
                ("a1", "2023/04/01 (Sat) 12:00", _session(EVIDENCE_TURN, "hm", answer_indices=(0,))),
                ("a2", "2023/04/02 (Sun) 12:00", _session(FILLER, "ok")),
            ],
            ["a1"],
        ),
        _item("q_tr_1", "temporal-reasoning",
              [("t1", "2023/03/01 (Wed) 07:15", _session(EVIDENCE_TURN, "yes", answer_indices=(0,)))], ["t1"],
              answer=42),
        _item("q_ssa_1", "single-session-assistant",
              [("sa1", "2023/02/01 (Wed) 07:15", _session("q", EVIDENCE_TURN, answer_indices=(1,)))], ["sa1"]),
        _item("q_ssp_1", "single-session-preference",
              [("sp1", "2023/01/01 (Sun) 07:15", _session(EVIDENCE_TURN, "ok", answer_indices=(0,)))], ["sp1"]),
        _item("q_ku_2", "knowledge-update",
              [("k1", "2023/01/05 (Thu) 07:15", _session(EVIDENCE_TURN, "ok", answer_indices=(0,)))], ["k1"]),
    ]
    # A 12-session item to exercise the SESSIONS_PER_ITEM cap: evidence
    # sits at positions 9 and 11, distractors fill from the front.
    big_sessions = []
    for i in range(12):
        marker = (9, 11)
        content = EVIDENCE_TURN if i in marker else f"{FILLER} (session {i})"
        big_sessions.append((
            f"big{i:02d}", f"2023/05/{i + 1:02d} (Mon) 10:00",
            _session(content, "ok", answer_indices=(0,) if i in marker else ()),
        ))
    items.append(_item("q_big_1", "multi-session", big_sessions, ["big09", "big11"]))
    return items


class FakeFetcher:
    def __init__(self, tmp_path, raw):
        self._path = tmp_path / "longmemeval_s_cleaned.json"
        self._path.write_text(json.dumps(raw))

    def fetch(self):
        return str(self._path)


@pytest.fixture
def loader(tmp_path):
    return LongMemEvalV1Loader(fetcher=FakeFetcher(tmp_path, _build_raw()))


class TestEvidenceMapping:
    def test_gold_evidence_is_has_answer_turn_contents(self, loader):
        item = next(i for i in loader.load() if i.id == "q_ssu_1")
        assert item.gold_evidence == [EVIDENCE_TURN, SECOND_EVIDENCE]

    def test_multi_session_evidence_collects_across_sessions(self, loader):
        item = next(i for i in loader.load() if i.id == "q_ms_1")
        assert item.gold_evidence == [EVIDENCE_TURN, SECOND_EVIDENCE]

    def test_has_answer_never_leaks_into_provider_messages(self, loader):
        for item in loader.load():
            for session in item.sessions:
                for message in session.messages:
                    assert set(message.keys()) == {"role", "content"}

    def test_abstention_item_has_no_gold_but_keeps_sessions(self, loader):
        item = next(i for i in loader.load() if i.id == "q_ku_1_abs")
        assert item.question_type == "abstention"
        assert item.answerable is False
        assert item.gold_evidence == []
        assert [s.session_id for s in item.sessions] == ["a1", "a2"]

    def test_int_answer_becomes_string(self, loader):
        item = next(i for i in loader.load() if i.id == "q_tr_1")
        assert item.gold_answer == "42"


class TestSessionsAndTimestamps:
    def test_timestamps_are_iso_utc(self, loader):
        item = next(i for i in loader.load() if i.id == "q_ssu_1")
        assert [s.timestamp for s in item.sessions] == [
            "2023-05-20T02:21:00Z", "2023-05-21T09:00:00Z", "2023-05-22T11:30:00Z",
        ]

    def test_namespace_is_per_item(self, loader):
        namespaces = {i.namespace for i in loader.load()}
        assert len(namespaces) == len(loader.load())
        assert all(ns.startswith("lme_v1_") for ns in namespaces)

    def test_to_iso_rejects_unexpected_format(self):
        with pytest.raises(ValueError):
            _to_iso("May 20, 2023 02:21")


class TestScopeCap:
    def test_cap_keeps_all_evidence_and_earliest_distractors(self, loader):
        item = next(i for i in loader.load() if i.id == "q_big_1")
        ids = [s.session_id for s in item.sessions]
        assert len(ids) == SESSIONS_PER_ITEM
        assert "big09" in ids and "big11" in ids
        assert ids == sorted(ids)  # chronological haystack order preserved
        assert ids[:6] == ["big00", "big01", "big02", "big03", "big04", "big05"]
        assert item.gold_evidence == [EVIDENCE_TURN, EVIDENCE_TURN]

    def test_small_items_are_not_padded(self, loader):
        item = next(i for i in loader.load() if i.id == "q_tr_1")
        assert len(item.sessions) == 1


class TestSamplingAndDigest:
    def test_full_load_is_sorted_by_id(self, loader):
        ids = [i.id for i in loader.load()]
        assert ids == sorted(ids)

    def test_seven_strata_are_represented(self, loader):
        strata = Counter(i.question_type for i in loader.load())
        assert strata["abstention"] == 1
        assert set(strata) == {
            "abstention", "multi-session", "single-session-user", "single-session-assistant",
            "single-session-preference", "temporal-reasoning", "knowledge-update",
        }

    def test_stratified_sample_returns_exact_size_deterministically(self, loader):
        first = loader.load(sample=5, seed=7)
        second = loader.load(sample=5, seed=7)
        assert len(first) == 5
        assert [i.id for i in first] == [i.id for i in second]

    def test_sha256_is_artifact_digest(self, loader, tmp_path):
        expected = hashlib.sha256((tmp_path / "longmemeval_s_cleaned.json").read_bytes()).hexdigest()
        assert loader.sha256() == expected


class _FakeStreamResponse:
    def __init__(self, payload: bytes):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        return None

    def iter_bytes(self, chunk_size):
        yield self._payload


class TestPinnedFileFetcher:
    PAYLOAD = b'{"good": true}'

    def _fetcher(self, tmp_path):
        return PinnedFileFetcher(
            url="https://origin.example/artifact.json",
            cache_path=tmp_path / "artifact.json",
            expected_sha256=hashlib.sha256(self.PAYLOAD).hexdigest(),
        )

    def test_valid_cache_is_served_without_network(self, tmp_path, monkeypatch):
        (tmp_path / "artifact.json").write_bytes(self.PAYLOAD)

        def no_network(*args, **kwargs):
            raise AssertionError("network must not be touched for a valid cache")

        monkeypatch.setattr("memarena.datasets._hf_fetch.httpx.stream", no_network)
        assert self._fetcher(tmp_path).fetch() == str(tmp_path / "artifact.json")

    def test_corrupted_cache_is_rehashed_and_redownloaded(self, tmp_path, monkeypatch):
        (tmp_path / "artifact.json").write_bytes(b"tampered bytes")
        monkeypatch.setattr(
            "memarena.datasets._hf_fetch.httpx.stream",
            lambda *a, **k: _FakeStreamResponse(self.PAYLOAD),
        )
        path = self._fetcher(tmp_path).fetch()
        assert (tmp_path / "artifact.json").read_bytes() == self.PAYLOAD
        assert path == str(tmp_path / "artifact.json")

    def test_download_that_fails_verification_raises_and_leaves_no_cache(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "memarena.datasets._hf_fetch.httpx.stream",
            lambda *a, **k: _FakeStreamResponse(b"wrong content from origin"),
        )
        with pytest.raises(ChecksumMismatchError):
            self._fetcher(tmp_path).fetch()
        assert not (tmp_path / "artifact.json").exists()
