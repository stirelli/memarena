"""Judge client tests (§5.8, Day 4): versioned instrument, JSON schema
validation, sqlite verdict cache keyed by (version, payload)."""


import pytest

from memarena.errors import ProviderError
from memarena.metrics.judge import Judge, JudgeCache, load_judges


def _judge(tmp_path, *, chat_fn, prompt="Grade it.\n", cache=None, name="answer_correctness",
           required=("correct",)):
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text(prompt)
    return Judge(name=name, model="test-model", prompt_path=prompt_path,
                 required_keys=required, chat_fn=chat_fn, cache=cache)


class TestVersioning:
    def test_version_pins_name_model_and_prompt_digest(self, tmp_path):
        judge = _judge(tmp_path, chat_fn=lambda s, u: '{"correct": true}')
        assert judge.version.startswith("answer_correctness@test-model:")
        assert len(judge.version.split(":")[1]) == 12

    def test_prompt_change_changes_version(self, tmp_path):
        a = _judge(tmp_path, chat_fn=lambda s, u: '{"correct": true}', prompt="v1")
        b = _judge(tmp_path, chat_fn=lambda s, u: '{"correct": true}', prompt="v2")
        assert a.version != b.version

    def test_verdict_carries_judge_version(self, tmp_path):
        judge = _judge(tmp_path, chat_fn=lambda s, u: '{"correct": false, "reasoning": "wrong number"}')
        verdict = judge.judge({"question": "q", "gold_answer": "4", "answer": "5"})
        assert verdict["correct"] is False
        assert verdict["judge_version"] == judge.version


class TestSchemaValidation:
    def test_missing_required_key_retries_then_raises(self, tmp_path):
        calls = []

        def bad_chat(s, u):
            calls.append(u)
            return '{"reasoning": "no verdict key"}'

        judge = _judge(tmp_path, chat_fn=bad_chat)
        with pytest.raises(ProviderError, match="invalid output twice"):
            judge.judge({"question": "q"})
        assert len(calls) == 2

    def test_malformed_json_retries_once_then_succeeds(self, tmp_path):
        replies = iter(["not json at all", '{"correct": true}'])
        judge = _judge(tmp_path, chat_fn=lambda s, u: next(replies))
        assert judge.judge({"question": "q"})["correct"] is True


class TestCache:
    def test_same_payload_hits_cache(self, tmp_path):
        calls = []

        def chat(s, u):
            calls.append(u)
            return '{"correct": true}'

        cache = JudgeCache(tmp_path / "cache.sqlite")
        judge = _judge(tmp_path, chat_fn=chat, cache=cache)
        payload = {"question": "q", "answer": "a"}
        first = judge.judge(payload)
        second = judge.judge(payload)
        assert first == second
        assert len(calls) == 1

    def test_version_change_misses_cache(self, tmp_path):
        calls = []

        def chat(s, u):
            calls.append(u)
            return '{"correct": true}'

        cache = JudgeCache(tmp_path / "cache.sqlite")
        payload = {"question": "q"}
        _judge(tmp_path, chat_fn=chat, prompt="v1", cache=cache).judge(payload)
        _judge(tmp_path, chat_fn=chat, prompt="v2", cache=cache).judge(payload)
        assert len(calls) == 2


class TestPayloadRendering:
    def test_lists_render_as_bullets_and_reach_the_judge(self, tmp_path):
        captured = {}

        def chat(s, u):
            captured["user"] = u
            return '{"covered": true}'

        judge = _judge(tmp_path, chat_fn=chat, name="evidence_coverage", required=("covered",))
        judge.judge({
            "question": "how many days in Japan?",
            "retrieved_memories": ["User visited Japan April 15-22", "User likes sushi"],
        })
        assert "### question" in captured["user"]
        assert "- User visited Japan April 15-22" in captured["user"]


class TestLoadJudges:
    def test_loads_all_three_judges_from_repo_config(self):
        judges = load_judges(chat_fn=lambda s, u: "{}")
        assert set(judges) == {"answer_correctness", "evidence_coverage", "abstention"}
        assert judges["evidence_coverage"].required_keys == ("covered",)
        # prompts are the repo files — versions must be stable strings
        assert all("@gpt-4.1:" in j.version for j in judges.values())
