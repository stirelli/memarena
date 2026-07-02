from __future__ import annotations

from pathlib import Path

import typer
from dotenv import load_dotenv

from memarena.config import RunConfig, load_yaml_dict
from memarena.registry import get_dataset_class, get_provider_class
from memarena.runner import RunResult
from memarena.runner import run as run_experiment

app = typer.Typer(add_completion=False)

PRICING_PATH = "configs/pricing.yaml"


@app.callback()
def main() -> None:
    """MemArena — the neutral benchmark harness for AI agent memory."""


@app.command("run")
def run_command(
    config: Path = typer.Option(..., "--config", help="Path to a run config YAML file (§Appendix B)."),  # noqa: B008
    provider: list[str] = typer.Option(  # noqa: B008
        None, "--provider",
        help="Run only these adapters from the config (repeatable). "
             "Enables per-provider sharding (CI matrix, parallel local runs).",
    ),
) -> None:
    """Run a memarena benchmark experiment (§5.3): every provider x dataset
    pair in the config, seeded and journaled, printing Level-1 metrics."""
    load_dotenv()
    run_config = RunConfig.from_yaml(config)
    if provider:
        known = {p.adapter for p in run_config.providers}
        unknown = [name for name in provider if name not in known]
        if unknown:
            raise typer.BadParameter(f"--provider {unknown} not in config (config has: {sorted(known)})")
        run_config.providers = [p for p in run_config.providers if p.adapter in set(provider)]
    pricing_table = load_yaml_dict(PRICING_PATH) if Path(PRICING_PATH).exists() else {}

    for dataset_section in run_config.datasets:
        dataset_cls = get_dataset_class(dataset_section.name)
        dataset_loader = dataset_cls()
        items = dataset_loader.load(
            sample=dataset_section.sample,
            seed=run_config.run.seed,
            stratify_by=dataset_section.stratify_by,
        )
        dataset_digest = dataset_loader.sha256()

        for provider_section in run_config.providers:
            provider_cls = get_provider_class(provider_section.adapter)
            provider_config = load_yaml_dict(provider_section.config)
            provider = provider_cls(provider_config)

            output_dir = Path(run_config.output.dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            journal_path = output_dir / f"{provider_section.adapter}__{dataset_section.name}__journal.jsonl"

            result = run_experiment(
                provider,
                items,
                run_id=run_config.run.id,
                seed=run_config.run.seed,
                dataset_digest=dataset_digest,
                repetitions=run_config.run.repetitions,
                top_k=provider_config.get("top_k", 5),
                budget_usd_max=run_config.run.budget_usd_max,
                pricing=pricing_table.get(provider_section.adapter),
                journal_path=journal_path,
            )

            _print_result(provider_section.adapter, dataset_section.name, result)


def _fmt(value: float | None, spec: str = ".3f") -> str:
    return "N/A" if value is None else format(value, spec)


def _print_result(provider_name: str, dataset_name: str, result: RunResult) -> None:
    metrics = result.metrics
    typer.echo(f"=== {provider_name} on {dataset_name} ===")
    typer.echo(f"Recall@5: {_fmt(metrics.recall_at_k.get(5))}")
    typer.echo(f"NDCG@5: {_fmt(metrics.ndcg_at_k.get(5))}")
    typer.echo(f"MRR: {_fmt(metrics.mrr)}")
    typer.echo(
        f"Add latency p50/p95 (ms): {_fmt(metrics.add_latency_p50_ms, '.1f')} / "
        f"{_fmt(metrics.add_latency_p95_ms, '.1f')}"
    )
    typer.echo(
        f"Search latency p50/p95 (ms): {_fmt(metrics.search_latency_p50_ms, '.1f')} / "
        f"{_fmt(metrics.search_latency_p95_ms, '.1f')}"
    )
    budget_note = "truncated" if result.budget_truncated else "not truncated"
    typer.echo(f"Cost: ${result.total_cost_usd:.4f} (budget: {budget_note})")
    typer.echo(f"Items: {metrics.n_scored_items} scored ({result.infra_error_count} infra errors)")
