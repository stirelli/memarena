import pytest

from memarena.answering import ABSTENTION_MARKER, ReaderAnswer, answer_question, chronological
from memarena.errors import ProviderError
from memarena.providers.base import MemoryRecord


def _rec(content, created_at=None, id="r"):
    return MemoryRecord(id=id, content=content, created_at=created_at)


class TestAnswerQuestion:
    def test_answers_using_retrieved_context(self):
        def chat_fn(system_prompt, user_prompt):
            assert "Biscuit" in user_prompt
            return "Biscuit"

        result = answer_question("What is my dog's name?", [_rec("My dog's name is Biscuit.")], chat_fn=chat_fn)
        assert isinstance(result, ReaderAnswer)
        assert result.text == "Biscuit"
        assert result.abstained is False

    def test_abstention_marker_sets_abstained_true(self):
        def chat_fn(system_prompt, user_prompt):
            return ABSTENTION_MARKER

        result = answer_question("What is my blood type?", [], chat_fn=chat_fn)
        assert result.abstained is True
        assert result.text == ABSTENTION_MARKER

    def test_empty_retrieved_context_still_calls_reader(self):
        calls = []

        def chat_fn(system_prompt, user_prompt):
            calls.append(user_prompt)
            return ABSTENTION_MARKER

        answer_question("anything?", [], chat_fn=chat_fn)
        assert len(calls) == 1

    def test_chat_fn_failure_raises_provider_error(self):
        def bad_chat_fn(system_prompt, user_prompt):
            raise RuntimeError("network down")

        with pytest.raises(ProviderError):
            answer_question("q?", [_rec("ctx")], chat_fn=bad_chat_fn)

    def test_system_prompt_is_abstention_aware(self):
        captured = {}

        def chat_fn(system_prompt, user_prompt):
            captured["system"] = system_prompt
            return "42"

        answer_question("q?", [_rec("ctx")], chat_fn=chat_fn)
        assert ABSTENTION_MARKER in captured["system"]


class TestChronologicalOrdering:
    """LongMemEval paper protocol: the reader sees memories in chronological
    order, not relevance order; timestamps are shown in the context."""

    def test_sorts_by_created_at_ascending(self):
        records = [
            _rec("newest", "2023-06-01T00:00:00Z", id="c"),
            _rec("oldest", "2023-01-01T00:00:00Z", id="a"),
            _rec("middle", "2023-03-01T00:00:00Z", id="b"),
        ]
        assert [r.content for r in chronological(records)] == ["oldest", "middle", "newest"]

    def test_undated_records_keep_retrieval_order_after_dated(self):
        records = [
            _rec("undated-1"),
            _rec("dated", "2023-01-01T00:00:00Z"),
            _rec("undated-2"),
        ]
        assert [r.content for r in chronological(records)] == ["dated", "undated-1", "undated-2"]

    def test_unparseable_timestamp_counts_as_undated(self):
        records = [
            _rec("garbage-date", "not a timestamp"),
            _rec("dated", "2023-01-01T00:00:00Z"),
        ]
        assert [r.content for r in chronological(records)] == ["dated", "garbage-date"]

    def test_mixed_naive_and_aware_timestamps_do_not_crash(self):
        records = [
            _rec("aware", "2023-05-01T00:00:00+00:00"),
            _rec("naive", "2023-04-01T00:00:00"),
        ]
        assert [r.content for r in chronological(records)] == ["naive", "aware"]

    def test_ties_keep_retrieval_order(self):
        records = [
            _rec("first-retrieved", "2023-01-01T00:00:00Z"),
            _rec("second-retrieved", "2023-01-01T00:00:00Z"),
        ]
        assert [r.content for r in chronological(records)] == ["first-retrieved", "second-retrieved"]

    def test_reader_context_is_chronological_with_timestamps(self):
        captured = {}

        def chat_fn(system_prompt, user_prompt):
            captured["user"] = user_prompt
            return "ok"

        records = [
            _rec("second event", "2023-06-01T00:00:00Z"),
            _rec("first event", "2023-01-01T00:00:00Z"),
        ]
        answer_question("q?", records, chat_fn=chat_fn)
        prompt = captured["user"]
        assert prompt.index("first event") < prompt.index("second event")
        assert "[2023-01-01T00:00:00Z]" in prompt
