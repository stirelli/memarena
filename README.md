# MemArena

**The neutral benchmark harness and public leaderboard for AI agent memory.**

Dozens of agent-memory systems claim to be #1 on the same public benchmarks.
MemArena is the arbiter: it plugs any memory system in behind a minimal
`MemoryProvider` interface, runs the public benchmarks with seeded
reproducibility, and reports accuracy with cost and latency attached.
Vendors can pay for testing. Vendors cannot pay for results.

Status: **pre-launch, Day 3 build.** Implemented so far: the harness runner
(seeded, journaled, budget-guarded), four providers (`baseline_rag`, `mem0`,
`zep`, `letta`, client versions pinned), two dataset loaders (LongMemEval V1
as the primary retrieval-quality track, LongMemEval-V2 as the latency/cost
track), and Level-1 deterministic metrics (verbatim evidence recall@k,
verbatim NDCG@k, verbatim MRR, latency percentiles, metered cost). Verbatim
metrics apply to extractive stores only; abstractive stores (distilled
memories) report N/A and are measured by a judged evidence-coverage metric
from Day 4 on (see docs/METHODOLOGY_NOTES.md). LLM-judge calibration and
the public leaderboard land in the days ahead.

## Quickstart

```bash
pip install -e ".[dev]"
cp .env.example .env  # add OPENAI_API_KEY (plus MEM0/ZEP/LETTA keys for those adapters)
memarena run --config configs/smoke.yaml
```

This runs the `baseline_rag` provider against a 20-item synthetic smoke
dataset and prints Recall@5, NDCG@5, MRR, and add/search latency.

Benchmark runs shard per provider, locally
(`memarena run --config configs/run.day3_v1.yaml --provider zep`) or on
GitHub-hosted runners (`.github/workflows/benchmark.yml`, manual trigger,
keys from repo secrets, journals uploaded as artifacts).

## Latency semantics

Add latency is reported per provider under one of two documented semantics
(see `providers/base.py`): time-to-settled (baseline_rag, mem0, letta: the
write is queryable when `add()` returns) or time-to-accepted plus a separate
journaled `settle_latency_ms` (zep, whose ingestion pipeline is asynchronous
by design). The two are different quantities and are never compared as one
number.

## Development

```bash
pip install -e ".[dev]"
ruff check .
pytest
```

## License

Apache 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
