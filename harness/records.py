"""What a run is made of: its header, its steps, and how it ended.

These are plain values. ``harness.loop`` produces them and ``harness.trajectory`` stores them. How a
run ends and how the world judges it are in ``harness.outcomes``, which needs no agent stack.
"""

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from harness.chat import Reply
from harness.outcomes import Outcome, SafetyVerdict, Verdict
from harness.repl import Observation

FINGERPRINT_LENGTH = 8


@dataclass(frozen=True)
class AgentPrompt:
    """What one agent was told before the task."""

    agent: str | None
    """The agent's name in a team; ``None`` in a run with one agent."""
    world_document: str
    """The project's instructions and the agent's tools: what any harness would be told about the world."""
    protocol: str
    """This harness's own part of the prompt: the cell format, ``show``, ``done``, the limits, the team."""

    @property
    def system_prompt(self) -> str:
        """The system message as sent: the world document, a blank line, the protocol."""
        return f"{self.world_document}\n\n{self.protocol}"


@dataclass(frozen=True)
class RunHeader:
    task: str
    prompts: tuple[AgentPrompt, ...]
    """One per agent, in the order they take turns. A run with one agent has one."""
    recipe: dict[str, Any]
    """Everything that changes what the model sees or says: agent, REPL, and chat settings."""
    initial_state: dict[str, Any]
    """What rebuilds the world as the run found it: a seed, a scene's name."""
    experiment: str | None
    """The slug of the ``docs/experiments`` file this run answers, if any."""

    @property
    def fingerprint(self) -> str:
        """A short hash of the recipe and system prompt: equal across runs of one setup.

        The task and the initial state are left out, so runs to pool into one rate share it.
        """
        prompts = [prompt.system_prompt for prompt in self.prompts]
        prompt = prompts[0] if len(prompts) == 1 else prompts  # one agent: the hash it always had
        canonical = json.dumps({"recipe": self.recipe, "system_prompt": prompt}, sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()[:FINGERPRINT_LENGTH]


@dataclass(frozen=True)
class Step:
    index: int
    reply: Reply
    code: str | None
    """The cell that ran; ``None`` when the reply held no code block."""
    observation: Observation
    variables: str
    """The summary of its variables the model was shown with the observation."""
    started: float
    """Unix time the model was asked for this step."""
    cell_seconds: float
    """Wall time the cell ran."""
    elapsed: float
    """Seconds from the start of the run to the end of this step."""
    time_left: float | None
    """Seconds of the run's time limit left at the end of this step, as the model was told; ``None`` without a limit."""
    agent: str | None
    """Which agent of a team took this step; ``None`` in a run with one agent."""
    heard: tuple[str, ...]
    """What the other agents had sent this one since its last turn, as it was shown before it replied."""
    told: tuple[str, ...]
    """What this step's cell sent to the other agents with ``tell()``."""


@dataclass(frozen=True)
class RunResult:
    outcome: Outcome
    answer: str | None
    """What the model passed to the call that ended the run. A claim: ``verdict`` says whether it holds."""
    steps: int
    verdict: Verdict | None
    """``None`` when nothing was graded: the run was interrupted or crashed, or its grader declined."""
    safety: SafetyVerdict | None
    """``None`` when the task defines no harm, or the run was interrupted or crashed."""
