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

    def test_known_token_count_hand_computed(self):
        # 1008 chars at the documented 4 chars/token = 252 tokens;
        # at $1.00 per 1k tokens the cost is exactly $0.252.
        assert estimate_cost_usd(1008, usd_per_1k_tokens=1.0) == pytest.approx(0.252)


class TestCostMeter:
    def test_run_cost_matches_hand_computed_char_count(self, tmp_path):
        # Ingested content: 1000 chars. Question: 8 chars. Total 1008 chars
        # = 252 tokens at 4 chars/token -> $0.252 at $1.00/1k tokens.
        items = [_item("i1", "ns1", "a" * 1000, "abcdefgh", [])]
        result = run(
            FakeProvider(), items, run_id="t", seed=42, repetitions=1, top_k=5,
            budget_usd_max=None, pricing={"usd_per_1k_tokens": 1.0},
            journal_path=tmp_path / "j.jsonl", dataset_digest="d",
        )
        assert result.total_cost_usd == pytest.approx(0.252)

    def test_cached_ingestion_charges_only_the_question(self, tmp_path):
        # Second item shares the namespace: ingestion is cached, so only its
        # 8-char question (2 tokens = $0.002) is added on top of $0.252.
        items = [
            _item("i1", "shared", "a" * 1000, "abcdefgh", []),
            _item("i2", "shared", "a" * 1000, "abcdefgh", []),
        ]
        result = run(
            FakeProvider(), items, run_id="t", seed=42, repetitions=1, top_k=5,
            budget_usd_max=None, pricing={"usd_per_1k_tokens": 1.0},
            journal_path=tmp_path / "j.jsonl", dataset_digest="d",
        )
        assert result.total_cost_usd == pytest.approx(0.254)


class _FakeClock:
    def __init__(self):
        self.now = 0.0

    def perf_counter(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class _InstrumentedProvider(FakeProvider):
    """Advances a fake clock by a fixed amount per operation so the runner's
    latency windows can be asserted exactly, with no real sleeps."""

    def __init__(self, clock: _FakeClock, *, reset_s: float, add_s: float, search_s: float):
        super().__init__()
        self._clock = clock
        self._reset_s, self._add_s, self._search_s = reset_s, add_s, search_s

    def reset(self, namespace):
        self._clock.advance(self._reset_s)
        super().reset(namespace)

    def add(self, namespace, messages, *, session_id, timestamp):
        self._clock.advance(self._add_s)
        super().add(namespace, messages, session_id=session_id, timestamp=timestamp)

    def search(self, namespace, query, *, top_k=5):
        self._clock.advance(self._search_s)
        return super().search(namespace, query, top_k=top_k)


class TestLatencyWindows:
    def test_add_latency_excludes_reset_and_search_latency_is_search_only(self, monkeypatch, tmp_path):
        # Spec Appendix A / §5.7: latency is wall-clock around add/search,
        # measured by the runner. reset() is namespace hygiene, not add cost.
        # reset takes 5s, the single add takes 1s, search takes 0.5s:
        # add_latency must be 1000ms (not 6000ms), search_latency 500ms.
        import memarena.runner as runner_module

        clock = _FakeClock()
        monkeypatch.setattr(runner_module.time, "perf_counter", clock.perf_counter)
        provider = _InstrumentedProvider(clock, reset_s=5.0, add_s=1.0, search_s=0.5)
        items = [_item("i1", "ns1", "the sky is blue", "sky", ["the sky is blue"])]
        journal_path = tmp_path / "j.jsonl"

        run(
            provider, items, run_id="t", seed=42, repetitions=1, top_k=5,
            budget_usd_max=None, pricing=None, journal_path=journal_path,
            dataset_digest="d",
        )

        record = json.loads(journal_path.read_text().strip())
        assert record["add_latency_ms"] == pytest.approx(1000.0)
        assert record["search_latency_ms"] == pytest.approx(500.0)

    def test_settle_is_timed_separately_from_add_and_search(self, monkeypatch, tmp_path):
        # Accept-only providers (zep) finish ingestion in settle(); its time
        # must appear in settle_latency_ms only, never inside add or search
        # (§8 Day 3 latency-semantics contract in providers/base.py).
        import memarena.runner as runner_module

        clock = _FakeClock()
        monkeypatch.setattr(runner_module.time, "perf_counter", clock.perf_counter)

        class SettlingProvider(_InstrumentedProvider):
            def settle(self, namespace):
                self._clock.advance(7.0)

        provider = SettlingProvider(clock, reset_s=5.0, add_s=1.0, search_s=0.5)
        items = [_item("i1", "ns1", "the sky is blue", "sky", ["the sky is blue"])]
        journal_path = tmp_path / "j.jsonl"

        run(
            provider, items, run_id="t", seed=42, repetitions=1, top_k=5,
            budget_usd_max=None, pricing=None, journal_path=journal_path,
            dataset_digest="d",
        )

        record = json.loads(journal_path.read_text().strip())
        assert record["add_latency_ms"] == pytest.approx(1000.0)
        assert record["settle_latency_ms"] == pytest.approx(7000.0)
        assert record["search_latency_ms"] == pytest.approx(500.0)

    def test_cached_ingestion_has_null_settle_latency(self, tmp_path):
        items = [
            _item("i1", "shared", "fact A", "fact", ["fact A"]),
            _item("i2", "shared", "fact A", "fact", ["fact A"]),
        ]
        journal_path = tmp_path / "j.jsonl"
        run(
            FakeProvider(), items, run_id="t", seed=42, repetitions=1, top_k=5,
            budget_usd_max=None, pricing=None, journal_path=journal_path,
            dataset_digest="d",
        )
        lines = [json.loads(line) for line in journal_path.read_text().strip().splitlines()]
        assert lines[0]["settle_latency_ms"] is not None
        assert lines[1]["settle_latency_ms"] is None


class TestRecallKsCappedAtTopK:
    def test_ks_beyond_top_k_are_not_reported(self, tmp_path):
        # With top_k=3 only 3 records are ever requested, so "Recall@5" and
        # "Recall@10" would mislabel a top-3 measurement. Only k <= top_k
        # may be computed and journaled.
        items = [_item("i1", "ns1", "the sky is blue", "sky", ["the sky is blue"])]
        journal_path = tmp_path / "j.jsonl"

        result = run(
            FakeProvider(), items, run_id="t", seed=42, repetitions=1, top_k=3,
            budget_usd_max=None, pricing=None, journal_path=journal_path,
            dataset_digest="d",
        )

        assert set(result.metrics.recall_at_k) == {1, 3}
        record = json.loads(journal_path.read_text().strip())
        assert set(record["recall_at_k"]) == {"1", "3"}

    def test_default_top_k_5_reports_up_to_5(self, tmp_path):
        items = [_item("i1", "ns1", "the sky is blue", "sky", ["the sky is blue"])]
        result = run(
            FakeProvider(), items, run_id="t", seed=42, repetitions=1, top_k=5,
            budget_usd_max=None, pricing=None, journal_path=tmp_path / "j.jsonl",
            dataset_digest="d",
        )
        assert set(result.metrics.recall_at_k) == {1, 3, 5}


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
