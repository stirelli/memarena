import pytest

from memarena.datasets.base import QAItem
from memarena.datasets.longmemeval_v2 import LongMemEvalV2Loader

FIXTURES = "tests/fixtures"


class FakeFetcher:
    """Test double standing in for the real HF-download fetcher — returns
    fixture paths instead of hitting the network."""

    def __init__(self, questions_path, trajectories_path, haystack_path):
        self.questions_path = questions_path
        self.trajectories_path = trajectories_path
        self.haystack_path = haystack_path

    def fetch_questions(self) -> str:
        return self.questions_path

    def fetch_haystack_small(self) -> str:
        return self.haystack_path

    def fetch_trajectories(self, needed_ids: set[str]) -> str:
        return self.trajectories_path


@pytest.fixture
def loader(tmp_path):
    fetcher = FakeFetcher(
        f"{FIXTURES}/lme_v2_questions_sample.jsonl",
        f"{FIXTURES}/lme_v2_trajectories_sample.jsonl",
        f"{FIXTURES}/lme_v2_haystack_small_sample.json",
    )
    return LongMemEvalV2Loader(fetcher=fetcher, cache_dir=tmp_path)


class TestLongMemEvalV2Loader:
    def test_declares_metadata(self, loader):
        assert loader.name == "longmemeval_v2"
        assert loader.license == "Apache-2.0"
        assert loader.redistributable is False
        assert loader.revision == "f152293e235517d504809563c833d7190b8c713b"

    def test_load_returns_qaitems(self, loader):
        items = loader.load()
        assert all(isinstance(i, QAItem) for i in items)
        assert len(items) == 4

    def test_gold_evidence_is_always_empty(self, loader):
        for item in loader.load():
            assert item.gold_evidence == []

    def test_abs_question_types_are_not_answerable(self, loader):
        for item in loader.load():
            if item.question_type.endswith("-abs"):
                assert item.answerable is False
            else:
                assert item.answerable is True

    def test_namespace_is_shared_per_domain(self, loader):
        items = loader.load()
        namespaces = {item.namespace for item in items}
        assert namespaces == {"lme_v2_web", "lme_v2_enterprise"}

    def test_items_in_same_domain_share_identical_sessions(self, loader):
        items = loader.load()
        web_items = [i for i in items if i.namespace == "lme_v2_web"]
        assert len(web_items) == 2
        assert web_items[0].sessions == web_items[1].sessions

    def test_sessions_carry_capped_trajectory_content(self, loader):
        items = loader.load()
        item = items[0]
        assert len(item.sessions) == 2  # traj-e1, traj-e2 or traj-w1, traj-w2
        for session in item.sessions:
            assert session.messages[0]["role"] == "user"
            assert session.messages[0]["content"].startswith("Goal:")
            for message in session.messages[1:]:
                assert message["role"] == "assistant"

    def test_sample_is_deterministic_given_seed(self, loader):
        a = loader.load(sample=2, seed=7)
        b = loader.load(sample=2, seed=7)
        assert [i.id for i in a] == [i.id for i in b]
        assert len(a) == 2

    def test_sha256_is_stable_hex_digest(self, loader):
        digest_a = loader.sha256()
        digest_b = loader.sha256()
        assert digest_a == digest_b
        assert len(digest_a) == 64
        int(digest_a, 16)
