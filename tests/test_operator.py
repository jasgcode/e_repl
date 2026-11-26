"""The operator's verdict, with a scripted person at the prompt."""

from harness.operator import NOTE_PROMPT, QUESTION, operator_verdict
from harness.outcomes import Verdict


def _person(*answers: str):
    """An ``ask`` that types ``answers`` in order, and the prompts it was shown."""
    replies, prompts = iter(answers), []

    def ask(prompt: str) -> str:
        prompts.append(prompt)
        return next(replies)

    return ask, prompts


def test_yes_with_a_note_is_a_success_that_carries_the_note():
    ask, prompts = _person("Y", " the block is in the bin ")
    assert operator_verdict(ask) == Verdict(True, "operator: yes; the block is in the bin")
    assert prompts == [QUESTION, NOTE_PROMPT]


def test_no_without_a_note_is_a_failure():
    ask, _ = _person("no", "")
    assert operator_verdict(ask) == Verdict(False, "operator: no")


def test_a_skip_gives_no_verdict_and_asks_for_no_note():
    ask, prompts = _person("s")
    assert operator_verdict(ask) is None
    assert prompts == [QUESTION]


def test_an_answer_that_is_none_of_the_three_is_asked_again():
    ask, prompts = _person("", "maybe", "n", "")
    assert operator_verdict(ask) == Verdict(False, "operator: no")
    assert prompts == [QUESTION, QUESTION, QUESTION, NOTE_PROMPT]
