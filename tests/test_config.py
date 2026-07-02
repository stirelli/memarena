import pytest
import yaml
from pydantic import ValidationError

from memarena.config import RunConfig, load_yaml_dict


class TestRunConfig:
    def test_parses_smoke_config(self):
        config = RunConfig.from_yaml("configs/smoke.yaml")
        assert config.run.id == "smoke-v1"
        assert config.run.seed == 42
        assert config.run.repetitions == 1
        assert config.run.budget_usd_max == 1.0
        assert config.datasets[0].name == "smoke"
        assert config.datasets[0].sample is None
        assert config.providers[0].adapter == "baseline_rag"
        assert config.providers[0].config == "configs/providers/baseline.yaml"
        assert config.output.dir == "results/smoke"

    def test_defaults_apply(self, tmp_path):
        minimal = {
            "run": {"id": "r1"},
            "datasets": [{"name": "smoke"}],
            "providers": [{"adapter": "baseline_rag", "config": "x.yaml"}],
            "output": {"dir": "results/r1"},
        }
        path = tmp_path / "minimal.yaml"
        path.write_text(yaml.dump(minimal))

        config = RunConfig.from_yaml(path)
        assert config.run.seed == 42
        assert config.run.repetitions == 1
        assert config.run.budget_usd_max is None
        assert config.datasets[0].sample is None
        assert config.datasets[0].stratify_by == "question_type"

    def test_missing_required_field_raises(self, tmp_path):
        invalid = {"run": {"id": "r1"}, "datasets": [], "providers": []}  # missing 'output'
        path = tmp_path / "invalid.yaml"
        path.write_text(yaml.dump(invalid))

        with pytest.raises(ValidationError):
            RunConfig.from_yaml(path)


class TestLoadYamlDict:
    def test_loads_provider_config(self):
        data = load_yaml_dict("configs/providers/baseline.yaml")
        assert data["embedding_model"] == "text-embedding-3-small"
        assert data["chunk_size"] == 200

    def test_loads_pricing_config(self):
        data = load_yaml_dict("configs/pricing.yaml")
        assert data["baseline_rag"]["usd_per_1k_tokens"] == 0.00002
