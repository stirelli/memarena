from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

from memarena.errors import ProviderError
from memarena.providers.base import MemoryRecord

READER_MODEL = "gpt-5-mini-2025-08-07"  # pinned constant reader (§5.0.1) — confirmed available 2026-07-01
ABSTENTION_MARKER = "I don't know"

ChatFn = Callable[[str, str], str]  # (system_prompt, user_prompt) -> reply text

SYSTEM_PROMPT = (
    "You are answering a question using ONLY the context provided below. "
    "Do not use outside knowledge. If the context does not contain enough "
    f"information to answer confidently, reply with exactly: {ABSTENTION_MARKER}"
)


@dataclass(frozen=True)
class ReaderAnswer:
    text: str
    abstained: bool


def _default_chat_fn(model: str) -> ChatFn:
    def chat(system_prompt: str, user_prompt: str) -> str:
        import os

        import httpx
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ProviderError("OPENAI_API_KEY is not set; the answering layer needs it.")
        response = httpx.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0,
            },
            timeout=30.0,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()

    return chat


def _parse_timestamp(created_at: str | None) -> datetime | None:
    if not created_at:
        return None
    try:
        parsed = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    except ValueError:
        return None
    # Naive timestamps are treated as UTC so mixed aware/naive sets stay comparable.
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def chronological(records: list[MemoryRecord]) -> list[MemoryRecord]:
    """LongMemEval paper protocol: the reader sees retrieved memories in
    chronological order, not relevance order. Records with a parseable
    ISO-8601 created_at sort ascending by it (ties keep retrieval order);
    records without one keep their retrieval order after the dated ones."""
    dated = [(ts, i, r) for i, r in enumerate(records) if (ts := _parse_timestamp(r.created_at)) is not None]
    undated = [r for r in records if _parse_timestamp(r.created_at) is None]
    return [r for _, _, r in sorted(dated, key=lambda t: (t[0], t[1]))] + undated


def _context_line(record: MemoryRecord) -> str:
    if _parse_timestamp(record.created_at) is not None:
        return f"- [{record.created_at}] {record.content}"
    return f"- {record.content}"


def answer_question(question: str, retrieved: list[MemoryRecord], *,
                     model: str = READER_MODEL, chat_fn: ChatFn | None = None) -> ReaderAnswer:
    """Constant-reader answering layer (§5.0.1, §8 Day 2): ONE pinned reader
    model, abstention-aware prompt. Retrieved memories are re-ordered
    chronologically (with their timestamps shown) before the reader — the
    LongMemEval paper's reading protocol (§8 Day 3). Not wired into the
    Level-1 CLI runs; Day 4 (judge work) calls this against the calibrated
    grader."""
    fn = chat_fn or _default_chat_fn(model)
    ordered = chronological(retrieved)
    context = "\n".join(_context_line(r) for r in ordered) if ordered else "(no context retrieved)"
    user_prompt = f"Context:\n{context}\n\nQuestion: {question}"
    try:
        text = fn(SYSTEM_PROMPT, user_prompt)
    except ProviderError:
        raise
    except Exception as exc:
        raise ProviderError(f"answering layer reader call failed: {exc}") from exc
    return ReaderAnswer(text=text, abstained=text.strip() == ABSTENTION_MARKER)
