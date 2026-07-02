import hashlib
import re
from pathlib import Path

import yaml
from typer.testing import CliRunner

from memarena.cli import app

runner = CliRunner()
DIM = 32


def _bow_vector(text: str) -> list[float]:
    vector = [0.0] * DIM
    for word in re.findall(r"\w+", text.lower()):
        idx = int(hashlib.sha256(word.encode()).hexdigest(), 16) % DIM
        vector[idx] += 1.0
    return vector


class _FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return self._payload


def _patch_openai_embeddings(monkeypatch) -> None:
    import httpx

    def fake_post(url, *, headers=None, json=None, timeout=None):
        texts = json["input"]
        return _FakeResponse({"data": [{"embedding": _bow_vector(t)} for t in texts]})

    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-real")


def _write_smoke_config(tmp_path: Path, output_dir: Path, adapter: str = "baseline_rag") -> Path:
    config = yaml.safe_load(Path("configs/smoke.yaml").read_text())
    config["output"]["dir"] = str(output_dir)
    config["providers"][0]["adapter"] = adapter
    config_path = tmp_path / "smoke.yaml"
    config_path.write_text(yaml.dump(config))
    return config_path


class TestRunCommand:
    def test_prints_recall_mrr_and_latency_for_baseline_on_smoke(self, monkeypatch, tmp_path):
        _patch_openai_embeddings(monkeypatch)
        config_path = _write_smoke_config(tmp_path, tmp_path / "results")

        result = runner.invoke(app, ["run", "--config", str(config_path)])

        assert result.exit_code == 0, result.output
        assert "baseline_rag" in result.output
        assert "Verbatim recall@5" in result.output
        assert "Verbatim MRR" in result.output
        assert "latency" in result.output.lower()
        assert (tmp_path / "results" / "baseline_rag__smoke__journal.jsonl").exists()

    def test_unknown_provider_adapter_fails_clearly(self, monkeypatch, tmp_path):
        _patch_openai_embeddings(monkeypatch)
        config_path = _write_smoke_config(tmp_path, tmp_path / "results", adapter="not_a_real_adapter")

        result = runner.invoke(app, ["run", "--config", str(config_path)])

        assert result.exit_code != 0


class TestProviderShardFilter:
    def _two_provider_config(self, tmp_path, output_dir):
        config = yaml.safe_load(Path("configs/smoke.yaml").read_text())
        config["output"]["dir"] = str(output_dir)
        config["providers"] = [
            {"adapter": "baseline_rag", "config": "configs/providers/baseline.yaml"},
            {"adapter": "mem0", "config": "configs/providers/mem0.default.yaml"},
        ]
        config_path = tmp_path / "two.yaml"
        config_path.write_text(yaml.dump(config))
        return config_path

    def test_provider_option_runs_only_the_selected_shard(self, monkeypatch, tmp_path):
        _patch_openai_embeddings(monkeypatch)
        config_path = self._two_provider_config(tmp_path, tmp_path / "results")

        result = runner.invoke(app, ["run", "--config", str(config_path), "--provider", "baseline_rag"])

        assert result.exit_code == 0, result.output
        assert (tmp_path / "results" / "baseline_rag__smoke__journal.jsonl").exists()
        assert not (tmp_path / "results" / "mem0__smoke__journal.jsonl").exists()

    def test_provider_option_not_in_config_fails_clearly(self, monkeypatch, tmp_path):
        _patch_openai_embeddings(monkeypatch)
        config_path = self._two_provider_config(tmp_path, tmp_path / "results")

        result = runner.invoke(app, ["run", "--config", str(config_path), "--provider", "zep"])

        assert result.exit_code != 0
        assert "not in config" in result.output


class TestPrintResultNASafety:
    def test_all_na_metrics_print_na_not_fabricated_zeros(self, capsys):
        # A run where every item infra-errored has no defined metric at all.
        # It must print N/A across the board and never a fabricated 0.0
        # latency (review finding F5).
        from memarena.cli import _print_result
        from memarena.metrics.deterministic import RunMetrics
        from memarena.runner import RunResult

        metrics = RunMetrics(
            verbatim_recall_at_k={1: None, 3: None, 5: None},
            verbatim_ndcg_at_k={1: None, 3: None, 5: None},
            verbatim_mrr=None,
            add_latency_p50_ms=None,
            add_latency_p95_ms=None,
            search_latency_p50_ms=None,
            search_latency_p95_ms=None,
            n_items=0,
            n_scored_items=0,
        )
        result = RunResult(
            run_id="r", seed=1, metrics=metrics, total_cost_usd=0.0,
            budget_truncated=False, infra_error_count=3, n_items_attempted=3,
        )

        _print_result("some_provider", "some_dataset", result)

        out = capsys.readouterr().out
        assert "Verbatim recall@5: N/A" in out
        assert "Verbatim NDCG@5: N/A" in out
        assert "Verbatim MRR: N/A" in out
        assert "Add latency p50/p95 (ms): N/A / N/A" in out
        assert "Search latency p50/p95 (ms): N/A / N/A" in out
        assert "0 scored (3 infra errors)" in out
