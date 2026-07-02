import pytest

from memarena.answering import ABSTENTION_MARKER, ReaderAnswer, answer_question
from memarena.errors import ProviderError


class TestAnswerQuestion:
    def test_answers_using_retrieved_context(self):
        def chat_fn(system_prompt, user_prompt):
            assert "Biscuit" in user_prompt
            return "Biscuit"

        result = answer_question("What is my dog's name?", ["My dog's name is Biscuit."], chat_fn=chat_fn)
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
            answer_question("q?", ["ctx"], chat_fn=bad_chat_fn)

    def test_system_prompt_is_abstention_aware(self):
        captured = {}

        def chat_fn(system_prompt, user_prompt):
            captured["system"] = system_prompt
            return "42"

        answer_question("q?", ["ctx"], chat_fn=chat_fn)
        assert ABSTENTION_MARKER in captured["system"]
