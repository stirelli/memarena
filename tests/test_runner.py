import json

import pytest

from memarena.cache import IngestionCache, ingestion_cache_key
from memarena.datasets.base import QAItem, Session
from memarena.errors import ProviderError
from memarena.providers.base import MemoryProvider, MemoryRecord, ProviderInfo
from memarena.runner import estimate_cost_usd, run


class FakeProvider(MemoryProvider):
    """Deterministic in-memory provider for runner tests — substring match
    instead of real embeddings, so orchestration logic is tested in
    isolation from baseline_rag's ranking behavior."""

    def __init__(self, *, fail_namespaces: set[str] | None = None):
        self.store: dict[str, list[str]] = {}
        self.fail_namespaces = fail_namespaces or set()

    def info(self) -> ProviderInfo:
        return ProviderInfo(name="fake", client_version="0.0.1",
                             config_digest="deadbeef", pricing_model="self_hosted")

    def reset(self, namespace: str) -> None:
        self.store[namespace] = []

    def add(self, namespace, messages, *, session_id, timestamp) -> None:
        if namespace in self.fail_namespaces:
            raise ProviderError("boom")
        for message in messages:
            self.store.setdefault(namespace, []).append(message["content"])

    def search(self, namespace, query, *, top_k=5):
        matches = [c for c in self.store.get(namespace, []) if query.lower() in c.lower()]
        return [MemoryRecord(id=f"{namespace}:{i}", content=c) for i, c in enumerate(matches)][:top_k]


def _item(id_, namespace, content, question, gold_evidence, question_type="single_session"):
    return QAItem(
        id=id_, namespace=namespace,
        sessions=[Session(session_id="s1", timestamp="2026-01-01T00:00:00Z",
                           messages=[{"role": "user", "content": content}])],
        question=question, gold_evidence=gold_evidence, question_type=question_type,
    )


class TestEstimateCostUsd:
    def test_scales_with_chars_and_price(self):
        assert estimate_cost_usd(4000, usd_per_1k_tokens=0.02) == pytest.approx(0.02)

    def test_zero_chars_is_zero_cost(self):
        assert estimate_cost_usd(0, usd_per_1k_tokens=0.02) == 0.0


class TestRun:
    def test_basic_run_computes_metrics_and_writes_journal(self, tmp_path):
        items = [
            _item("i1", "ns1", "the sky is blue", "sky", ["the sky is blue"]),
            _item("i2", "ns2", "grass is green", "purple elephants", ["grass is green"]),
        ]
        journal_path = tmp_path / "journal.jsonl"

        result = run(
            FakeProvider(), items, run_id="test-run", seed=42, repetitions=1,
            top_k=5, budget_usd_max=None, pricing=None, journal_path=journal_path,
            dataset_digest="test-dataset",
        )

        assert result.run_id == "test-run"
        assert result.seed == 42
        assert result.budget_truncated is False
        assert result.infra_error_count == 0
        assert result.metrics.n_items == 2
        assert result.metrics.recall_at_k[5] == 0.5  # i1 hits, i2 misses
        assert result.metrics.mrr == 0.5

        lines = journal_path.read_text().strip().splitlines()
        assert len(lines) == 2
        record = json.loads(lines[0])
        assert record["run_id"] == "test-run"
        assert record["seed"] == 42
        assert record["item_id"] == "i1"
        assert record["status"] == "ok"
        assert "recall_at_k" in record
        assert "add_latency_ms" in record
        assert "search_latency_ms" in record

    def test_budget_guard_truncates_run(self, tmp_path):
        items = [
            _item("i1", "ns1", "the sky is blue", "sky", ["the sky is blue"]),
            _item("i2", "ns2", "grass is green", "grass", ["grass is green"]),
        ]
        journal_path = tmp_path / "journal.jsonl"

        result = run(
            FakeProvider(), items, run_id="test-run", seed=42, repetitions=1,
            top_k=5, budget_usd_max=1.0, pricing={"usd_per_1k_tokens": 1_000_000},
            journal_path=journal_path, dataset_digest="test-dataset",
        )

        assert result.budget_truncated is True
        assert result.metrics.n_items == 1  # only the first item ran before the guard tripped
        lines = journal_path.read_text().strip().splitlines()
        assert len(lines) == 1

    def test_infra_error_is_excluded_from_metrics_but_counted(self, tmp_path):
        items = [
            _item("i1", "ns1", "the sky is blue", "sky", ["the sky is blue"]),
            _item("i2", "ns_error", "grass is green", "grass", ["grass is green"]),
        ]
        journal_path = tmp_path / "journal.jsonl"

        result = run(
            FakeProvider(fail_namespaces={"ns_error"}), items, run_id="test-run", seed=42,
            repetitions=1, top_k=5, budget_usd_max=None, pricing=None, journal_path=journal_path,
            dataset_digest="test-dataset",
        )

        assert result.infra_error_count == 1
        assert result.metrics.n_items == 1  # the failing item is excluded, not just unscored
        lines = journal_path.read_text().strip().splitlines()
        assert len(lines) == 2
        statuses = {json.loads(line)["item_id"]: json.loads(line)["status"] for line in lines}
        assert statuses == {"i1": "ok", "i2": "infra_error"}

    def test_repetitions_produce_one_journal_entry_each(self, tmp_path):
        items = [_item("i1", "ns1", "the sky is blue", "sky", ["the sky is blue"])]
        journal_path = tmp_path / "journal.jsonl"

        result = run(
            FakeProvider(), items, run_id="test-run", seed=42, repetitions=2,
            top_k=5, budget_usd_max=None, pricing=None, journal_path=journal_path,
            dataset_digest="test-dataset",
        )

        assert result.metrics.n_items == 2  # 1 item x 2 repetitions
        lines = journal_path.read_text().strip().splitlines()
        assert len(lines) == 2
        assert {json.loads(line)["rep"] for line in lines} == {0, 1}


class TestIngestionReuse:
    def test_second_item_sharing_namespace_reuses_ingestion(self, tmp_path):
        provider = FakeProvider()
        items = [
            _item("i1", "shared-ns", "fact A", "fact", ["fact A"]),
            _item("i2", "shared-ns", "fact A", "fact", ["fact A"]),
        ]
        journal_path = tmp_path / "journal.jsonl"

        result = run(
            provider, items, run_id="t", seed=42, repetitions=1, top_k=5,
            budget_usd_max=None, pricing=None, journal_path=journal_path,
            dataset_digest="dset1",
        )

        # only one add() worth of content in the store — reset+add ran once
        assert provider.store["shared-ns"] == ["fact A"]
        lines = [json.loads(line) for line in journal_path.read_text().strip().splitlines()]
        assert lines[0]["ingested"] is True
        assert lines[1]["ingested"] is False
        assert lines[1]["add_latency_ms"] is None
        assert result.metrics.n_items == 2  # both items still scored via search

    def test_fresh_ingest_forces_reingestion_every_item(self, tmp_path):
        provider = FakeProvider()
        items = [
            _item("i1", "shared-ns", "fact A", "fact", ["fact A"]),
            _item("i2", "shared-ns", "fact A", "fact", ["fact A"]),
        ]
        journal_path = tmp_path / "journal.jsonl"

        run(
            provider, items, run_id="t", seed=42, repetitions=1, top_k=5,
            budget_usd_max=None, pricing=None, journal_path=journal_path,
            dataset_digest="dset1", fresh_ingest=True,
        )
        lines = [json.loads(line) for line in journal_path.read_text().strip().splitlines()]
        assert lines[0]["ingested"] is True
        assert lines[1]["ingested"] is True

    def test_different_dataset_digest_does_not_reuse_across_separate_runs(self, tmp_path):
        provider = FakeProvider()
        items = [_item("i1", "shared-ns", "fact A", "fact", ["fact A"])]
        cache = IngestionCache()
        cache.mark_ingested(ingestion_cache_key(provider.info(), dataset_digest="other-dset", namespace="shared-ns"))

        result = run(
            provider, items, run_id="t", seed=42, repetitions=1, top_k=5,
            budget_usd_max=None, pricing=None, journal_path=tmp_path / "j.jsonl",
            dataset_digest="dset1", ingestion_cache=cache,
        )
        assert result.metrics.n_items == 1
        assert provider.store["shared-ns"] == ["fact A"]  # still ingested — different digest, cache miss
