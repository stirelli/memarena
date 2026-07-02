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
        assert "Recall@5" in result.output
        assert "MRR" in result.output
        assert "latency" in result.output.lower()
        assert (tmp_path / "results" / "baseline_rag__smoke__journal.jsonl").exists()

    def test_unknown_provider_adapter_fails_clearly(self, monkeypatch, tmp_path):
        _patch_openai_embeddings(monkeypatch)
        config_path = _write_smoke_config(tmp_path, tmp_path / "results", adapter="not_a_real_adapter")

        result = runner.invoke(app, ["run", "--config", str(config_path)])

        assert result.exit_code != 0
