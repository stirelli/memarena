from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Callable
from pathlib import Path

from memarena.errors import ProviderError

# A judge is a versioned instrument (§5.8): (model id, prompt sha256,
# temperature=0, JSON output schema). Any component change bumps the
# version string, which invalidates the cache and annotates every verdict.

ChatFn = Callable[[str, str], str]  # (system_prompt, user_prompt) -> reply text


def _default_chat_fn(model: str) -> ChatFn:
    def chat(system_prompt: str, user_prompt: str) -> str:
        import os

        import httpx
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ProviderError("OPENAI_API_KEY is not set; the judge needs it.")
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
                "response_format": {"type": "json_object"},
            },
            timeout=60.0,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    return chat


class JudgeCache:
    """sqlite verdict cache keyed by (judge version, payload digest) — risk
    R2: judged items are re-scored for free across reruns and prompt
    iterations never reuse stale verdicts (the version is in the key)."""

    def __init__(self, path: str | Path = ".cache/judge_cache.sqlite"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path))
        self._conn.execute("CREATE TABLE IF NOT EXISTS verdicts (key TEXT PRIMARY KEY, verdict TEXT NOT NULL)")

    def get(self, key: str) -> dict | None:
        row = self._conn.execute("SELECT verdict FROM verdicts WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key: str, verdict: dict) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO verdicts (key, verdict) VALUES (?, ?)", (key, json.dumps(verdict)),
        )
        self._conn.commit()


class Judge:
    """One judged metric = one Judge instance bound to a prompt file.

    version = name@model:prompt_sha12 — recorded next to every verdict a
    caller stores, so published numbers always cite their instrument."""

    def __init__(self, *, name: str, model: str, prompt_path: str | Path,
                 required_keys: tuple[str, ...], chat_fn: ChatFn | None = None,
                 cache: JudgeCache | None = None):
        self.name = name
        self.model = model
        self.prompt_text = Path(prompt_path).read_text()
        self.required_keys = required_keys
        self._chat_fn = chat_fn or _default_chat_fn(model)
        self._cache = cache
        prompt_digest = hashlib.sha256(self.prompt_text.encode()).hexdigest()[:12]
        self.version = f"{name}@{model}:{prompt_digest}"

    def _render_payload(self, payload: dict) -> str:
        blocks = []
        for key, value in payload.items():
            if isinstance(value, list):
                rendered = "\n".join(f"- {v}" for v in value) if value else "(empty)"
            else:
                rendered = str(value)
            blocks.append(f"### {key}\n{rendered}")
        return "\n\n".join(blocks)

    def judge(self, payload: dict) -> dict:
        """Returns the verdict dict (validated against required_keys) with
        judge_version attached. Deterministic given (version, payload):
        temperature 0 plus the cache."""
        key = hashlib.sha256(f"{self.version}\n{json.dumps(payload, sort_keys=True)}".encode()).hexdigest()
        if self._cache is not None:
            cached = self._cache.get(key)
            if cached is not None:
                return cached

        user_prompt = self._render_payload(payload)
        last_error: Exception | None = None
        for _ in range(2):  # one retry on malformed output
            raw = self._chat_fn(self.prompt_text, user_prompt)
            try:
                verdict = json.loads(raw)
                missing = [k for k in self.required_keys if k not in verdict]
                if missing:
                    raise ValueError(f"verdict missing keys {missing}")
            except (json.JSONDecodeError, ValueError) as exc:
                last_error = exc
                continue
            verdict["judge_version"] = self.version
            if self._cache is not None:
                self._cache.put(key, verdict)
            return verdict
        raise ProviderError(f"judge {self.version} returned invalid output twice: {last_error}")


def load_judges(config_path: str | Path = "configs/judges/judge.v1.yaml",
                *, chat_fn: ChatFn | None = None, cache: JudgeCache | None = None) -> dict[str, Judge]:
    import yaml
    config = yaml.safe_load(Path(config_path).read_text())
    required = {
        "answer_correctness": ("correct",),
        "evidence_coverage": ("covered",),
        "abstention": ("abstained",),
    }
    return {
        name: Judge(
            name=name, model=config["model"], prompt_path=prompt_path,
            required_keys=required[name], chat_fn=chat_fn, cache=cache,
        )
        for name, prompt_path in config["judges"].items()
    }
