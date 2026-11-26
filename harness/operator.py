"""A person's verdict on a run, for a table no program can judge."""

from collections.abc import Callable

from harness.records import Verdict

QUESTION = "Did the robot do the task? [y]es, [n]o, or [s]kip to give no verdict"
NOTE_PROMPT = "Note for the record (optional)"
YES = ("y", "yes")
NO = ("n", "no")
SKIP = ("s", "skip")


def operator_verdict(ask: Callable[[str], str]) -> Verdict | None:
    """Ask the operator whether the task was done, then for a note. A skip gives no verdict.

    A run without a verdict is void: it is left out of a success rate. Skip when the run cannot be
    judged (the scene was disturbed, a camera dropped out), never because it went badly.

    Args:
        ask: Shows a prompt and returns the line the operator typed, which may be empty.
    """
    answer = ask(QUESTION).strip().lower()
    while answer not in (*YES, *NO, *SKIP):
        answer = ask(QUESTION).strip().lower()
    if answer in SKIP:
        return None
    note = ask(NOTE_PROMPT).strip()
    said = "yes" if answer in YES else "no"
    return Verdict(answer in YES, f"operator: {said}; {note}" if note else f"operator: {said}")
