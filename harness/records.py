"""What a run is made of: its header, its steps, and how it ended.

These are plain values. ``harness.loop`` produces them and ``harness.trajectory`` stores them.
"""

import hashlib
import json
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from harness.repl import Observation

FINGERPRINT_LENGTH = 8


class Outcome(StrEnum):
    DONE = "done"
    """The model called ``done()``."""
    STEP_LIMIT = "step_limit"
    INTERRUPTED = "interrupted"
    """Ctrl-C: on a robot, usually a person stopping a run that was going wrong."""
    CRASHED = "crashed"
    """An exception outside the model's code: the chat endpoint, the recorder, the harness."""


@dataclass(frozen=True)
class RunHeader:
    task: str
    system_prompt: str
    recipe: dict[str, Any]
    """Everything that changes what the model sees or says: agent, REPL, and chat settings."""
    experiment: str | None
    """The slug of the ``docs/experiments`` file this run answers, if any."""

    @property
    def fingerprint(self) -> str:
        """A short hash of the recipe and system prompt: equal across runs of one setup, whatever the task."""
        canonical = json.dumps({"recipe": self.recipe, "system_prompt": self.system_prompt}, sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()[:FINGERPRINT_LENGTH]


@dataclass(frozen=True)
class Step:
    index: int
    reply: str
    code: str | None
    """The cell that ran; ``None`` when the reply held no code block."""
    observation: Observation


@dataclass(frozen=True)
class RunResult:
    outcome: Outcome
    answer: str | None
    steps: int
