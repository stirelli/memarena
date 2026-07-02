# MemArena

**The neutral benchmark harness and public leaderboard for AI agent memory.**

Dozens of agent-memory systems claim to be #1 on the same public benchmarks.
MemArena is the arbiter: it plugs any memory system in behind a minimal
`MemoryProvider` interface, runs the public benchmarks with seeded
reproducibility, and reports accuracy with cost and latency attached.
Vendors can pay for testing. Vendors cannot pay for results.

Status: **pre-launch, Day 1 build.** The harness runner, baseline provider,
and Level-1 deterministic metrics are implemented; provider adapters,
LLM-judge calibration, and the public leaderboard land in the days ahead.

## Quickstart

```bash
pip install -e ".[dev]"
cp .env.example .env  # add OPENAI_API_KEY
memarena run --config configs/smoke.yaml
```

This runs the `baseline_rag` provider against a 20-item synthetic smoke
dataset and prints Recall@5, MRR, and add/search latency.

## Development

```bash
pip install -e ".[dev]"
ruff check .
pytest
```

## License

Apache 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
