"""Day 4: generate reader answers per provider over the Day 3 sample.

For every item: search the provider's store (top-k from its config), run
the pinned constant reader over the chronologically ordered memories, and
journal everything the judges and the labeling sheet need — including the
retrieved contents, which the runner's Level-1 journals deliberately do
not store.

Providers attach to the stores the Day 3 run left behind (letta: shared
cloud agent; mem0-oss: on-disk qdrant; zep-graphiti: on-disk kuzu).
baseline_rag is process-local, so it re-ingests before answering.

Run: .venv/bin/python scripts/day4_generate_answers.py [--provider NAME ...]
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from dotenv import load_dotenv

from memarena.answering import READER_MODEL, SYSTEM_PROMPT, answer_question
from memarena.config import load_yaml_dict
from memarena.datasets.longmemeval_v1 import LongMemEvalV1Loader
from memarena.errors import ProviderError
from memarena.registry import get_provider_class

OUT_DIR = Path("results/day4-answers")
SEED = 42
SAMPLE = 200

PROVIDER_CONFIGS = {
    "baseline_rag": "configs/providers/baseline.yaml",
    "mem0": "configs/providers/mem0.default.yaml",
    "zep": "configs/providers/zep.default.yaml",
    "letta": "configs/providers/letta.default.yaml",
}


def generate(provider_name: str, items) -> None:
    config = load_yaml_dict(PROVIDER_CONFIGS[provider_name])
    provider = get_provider_class(provider_name)(config)
    top_k = config.get("top_k", 5)
    prompt_digest = hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()[:12]

    if provider_name == "baseline_rag":  # in-memory store: rebuild before answering
        for i, item in enumerate(items, 1):
            provider.reset(item.namespace)
            for session in item.sessions:
                provider.add(item.namespace, session.messages,
                             session_id=session.session_id, timestamp=session.timestamp)
            if i % 25 == 0:
                print(f"  [baseline ingest] {i}/{len(items)}", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{provider_name}__answers.jsonl"
    n_errors = 0
    with out_path.open("w") as out:
        for i, item in enumerate(items, 1):
            row = {
                "item_id": item.id,
                "provider": provider_name,
                "memory_representation": provider.memory_representation,
                "question_type": item.question_type,
                "answerable": item.answerable,
                "question": item.question,
                "gold_answer": item.gold_answer,
                "gold_evidence": item.gold_evidence,
                "reader_model": READER_MODEL,
                "reader_prompt_digest": prompt_digest,
            }
            try:
                records = provider.search(item.namespace, item.question, top_k=top_k)
                answer = answer_question(item.question, records)
                row.update(
                    status="ok",
                    retrieved=[{"id": r.id, "content": r.content, "score": r.score,
                                "created_at": r.created_at} for r in records],
                    answer_text=answer.text,
                    abstained_marker=answer.abstained,
                )
            except ProviderError as exc:
                n_errors += 1
                row.update(status="error", error=str(exc))
            out.write(json.dumps(row) + "\n")
            out.flush()
            if i % 25 == 0:
                print(f"  [{provider_name}] {i}/{len(items)} answered", flush=True)
    print(f"[{provider_name}] done -> {out_path} ({n_errors} errors)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", action="append",
                        choices=sorted(PROVIDER_CONFIGS), default=None)
    args = parser.parse_args()
    providers = args.provider or ["baseline_rag", "mem0", "letta"]  # zep joins once its shard finishes

    load_dotenv()
    items = LongMemEvalV1Loader().load(sample=SAMPLE, seed=SEED)
    for name in providers:
        generate(name, items)


if __name__ == "__main__":
    main()
