from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel


def load_yaml_dict(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


class RunSection(BaseModel):
    id: str
    seed: int = 42
    repetitions: int = 1
    concurrency: int = 4
    budget_usd_max: float | None = None


class DatasetSection(BaseModel):
    name: str
    sample: int | None = None
    stratify_by: str | None = "question_type"


class ProviderSection(BaseModel):
    adapter: str
    config: str
    client_version: str | None = None


class OutputSection(BaseModel):
    dir: str
    otel_endpoint: str | None = None


class RunConfig(BaseModel):
    run: RunSection
    datasets: list[DatasetSection]
    providers: list[ProviderSection]
    output: OutputSection

    @classmethod
    def from_yaml(cls, path: str | Path) -> RunConfig:
        return cls.model_validate(load_yaml_dict(path))
