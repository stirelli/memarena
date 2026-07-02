from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from memarena.errors import ProviderError

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


def answer_question(question: str, retrieved_contents: list[str], *,
                     model: str = READER_MODEL, chat_fn: ChatFn | None = None) -> ReaderAnswer:
    """Constant-reader answering layer (§5.0.1, §8 Day 2): ONE pinned reader
    model, abstention-aware prompt. Not wired into the Day-2 CLI run — the
    Day 2 exit criterion is deterministic (Level-1) metrics only (§8); Day 4
    (judge work) calls this against the calibrated grader."""
    fn = chat_fn or _default_chat_fn(model)
    context = "\n".join(f"- {c}" for c in retrieved_contents) if retrieved_contents else "(no context retrieved)"
    user_prompt = f"Context:\n{context}\n\nQuestion: {question}"
    try:
        text = fn(SYSTEM_PROMPT, user_prompt)
    except ProviderError:
        raise
    except Exception as exc:
        raise ProviderError(f"answering layer reader call failed: {exc}") from exc
    return ReaderAnswer(text=text, abstained=text.strip() == ABSTENTION_MARKER)
