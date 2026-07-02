import pytest

from memarena.datasets.base import DatasetLoader, QAItem, Session
from memarena.datasets.smoke import SmokeDatasetLoader

STRATA = {"single_session", "multi_session", "knowledge_update", "temporal_reasoning", "abstention"}


def test_dataset_loader_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        DatasetLoader()


def test_session_and_qaitem_are_plain_dataclasses():
    session = Session(session_id="s1", timestamp="2026-01-01T00:00:00Z",
                       messages=[{"role": "user", "content": "hi"}])
    item = QAItem(
        id="x1", namespace="ns1", sessions=[session], question="q?",
        gold_evidence=["hi"], question_type="single_session",
    )
    assert item.answerable is True
    assert item.gold_answer is None


class TestSmokeDatasetLoader:
    def setup_method(self):
        self.loader = SmokeDatasetLoader()

    def test_declares_metadata(self):
        assert self.loader.name == "smoke"
        # self-authored synthetic set — unlike third-party datasets, it ships
        # freely under this repo's own Apache 2.0 license (§5.6).
        assert self.loader.redistributable is True

    def test_load_returns_twenty_items(self):
        items = self.loader.load()
        assert len(items) == 20
        assert all(isinstance(i, QAItem) for i in items)

    def test_ids_are_unique(self):
        items = self.loader.load()
        assert len({i.id for i in items}) == 20

    def test_stratified_four_per_question_type(self):
        items = self.loader.load()
        assert {i.question_type for i in items} == STRATA
        for stratum in STRATA:
            assert sum(1 for i in items if i.question_type == stratum) == 4

    def test_answerable_items_have_gold_evidence_present_verbatim_in_sessions(self):
        items = self.loader.load()
        for item in items:
            if not item.answerable:
                continue
            assert item.gold_evidence, f"{item.id} should have gold evidence"
            all_content = [
                m["content"] for s in item.sessions for m in s.messages
            ]
            for evidence in item.gold_evidence:
                assert evidence in all_content, f"{item.id}: evidence not found verbatim"

    def test_abstention_items_have_no_gold_evidence(self):
        items = self.loader.load()
        abstention_items = [i for i in items if i.question_type == "abstention"]
        assert len(abstention_items) == 4
        for item in abstention_items:
            assert item.answerable is False
            assert item.gold_evidence == []

    def test_multi_session_items_have_multiple_sessions(self):
        items = self.loader.load()
        multi = [i for i in items if i.question_type == "multi_session"]
        assert all(len(i.sessions) >= 2 for i in multi)

    def test_sample_is_deterministic_given_seed(self):
        a = self.loader.load(sample=10, seed=42)
        b = self.loader.load(sample=10, seed=42)
        assert [i.id for i in a] == [i.id for i in b]
        assert len(a) == 10

    def test_sha256_is_stable_hex_digest(self):
        digest_a = self.loader.sha256()
        digest_b = self.loader.sha256()
        assert digest_a == digest_b
        assert len(digest_a) == 64
        int(digest_a, 16)  # raises if not hex
