"""Run directories read back: what each run was, how it stands, where its time went.

``harness.trajectory`` writes run directories and this module reads them, by the same filenames.
The standard library only, so a script that counts rates or times runs needs neither the agent
stack nor the trace viewer, which renders what this module reads.
"""

import json
import math
import re
import statistics
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from harness.outcomes import DECIDED_LABELS, CellStatus, Outcome, SafetyLabel
from harness.replies import CODE_BLOCK, FinishReason, finish_reason

HEADER_FILENAME = "run.json"
STEPS_FILENAME = "steps.jsonl"
RESULT_FILENAME = "result.json"
IMAGE_MEDIA_TYPE = "image/jpeg"
IMAGE_FILE = r"step\d+_\d+\.jpg"
"""The names ``image_filename`` gives."""
WAITING_WORTH_A_FLAG = 0.2
"""A run that spent more than this share of its time waiting on the endpoint says little about the model's speed."""


def image_filename(step: int, image: int) -> str:
    return f"step{step:03d}_{image}.jpg"


class Standing(StrEnum):
    """Where a run stands with its grader."""

    SUCCESS = "success"
    FAILURE = "failure"
    VOID = "void"
    """Finished with no verdict: interrupted, crashed, or its grader declined."""
    UNFINISHED = "unfinished"
    """No ``result.json``: the run is being written, or its process was killed."""
    HARM_ONLY = "harm-only"
    """Graded for harm and not for success."""


def model_seconds(step: dict[str, Any]) -> float:
    """How long the endpoint took over the attempt that answered, in a ``steps.jsonl`` line."""
    return step.get("chat_request_seconds", step.get("chat_seconds", 0.0))


def step_finish_reason(step: dict[str, Any]) -> str | None:
    return finish_reason(step.get("response"))


def step_usage(step: dict[str, Any]) -> dict[str, Any]:
    """The endpoint's token counts for a step, or an empty dict when it gave none."""
    return (step.get("response") or {}).get("usage") or {}


@dataclass(frozen=True)
class Run:
    """One run directory, read as it is now."""

    name: str
    header: dict[str, Any]
    steps: tuple[dict[str, Any], ...]
    result: dict[str, Any] | None
    """``None`` when ``result.json`` is not there: the run is being written, or its process was killed."""
    changed: float
    """Unix time the run's record last changed."""

    @property
    def standing(self) -> Standing:
        if self.result is None:
            return Standing.UNFINISHED
        verdict = self.result.get("verdict")
        if verdict is None:
            return Standing.VOID if self.safety is None else Standing.HARM_ONLY
        return Standing.SUCCESS if verdict["success"] else Standing.FAILURE

    @property
    def outcome(self) -> str | None:
        """How the run ended, as ``Outcome`` names it; ``None`` before it has."""
        return (self.result or {}).get("outcome")

    @property
    def safety(self) -> dict[str, str] | None:
        """The run's safety verdict (``label``, ``reason``), or ``None`` when its task defines no harm."""
        return (self.result or {}).get("safety")

    @property
    def team(self) -> list[dict[str, Any]]:
        """The members of a team run, each with its ``name``, ``role`` and ``chat`` recipe; empty for one agent."""
        return self.header.get("recipe", {}).get("team") or []

    @property
    def model(self) -> str:
        """The model of a run with one agent, or each member's by name for a team."""
        if self.team:
            return ", ".join(f"{member['name']}: {member['chat'].get('model', '')}" for member in self.team)
        return str(self.header.get("recipe", {}).get("chat", {}).get("model", ""))

    @property
    def time_limit(self) -> float | None:
        return self._agent_recipe.get("time_limit")

    @property
    def max_steps(self) -> int | None:
        return self._agent_recipe.get("max_steps")

    @property
    def _agent_recipe(self) -> dict[str, Any]:
        return self.header.get("recipe", {}).get("agent", {})


def read_run(run_dir: Path) -> Run:
    steps_path, result_path = run_dir / STEPS_FILENAME, run_dir / RESULT_FILENAME
    lines = steps_path.read_text().splitlines() if steps_path.exists() else []
    steps = []
    for number, line in enumerate(lines, 1):
        try:
            steps.append(json.loads(line))
        except json.JSONDecodeError:
            if number < len(lines):
                raise
            # The last line of a run that is being written can be half there.
    # A step's images are written before its line, so these three files say when the run last changed.
    record = [path for path in (run_dir / HEADER_FILENAME, steps_path, result_path) if path.exists()]
    return Run(
        name=run_dir.name,
        header=json.loads((run_dir / HEADER_FILENAME).read_text()),
        steps=tuple(steps),
        result=json.loads(result_path.read_text()) if result_path.exists() else None,
        changed=max(path.stat().st_mtime for path in record),
    )


def run_names(runs_dir: Path) -> list[str]:
    """The run directories under ``runs_dir``, newest first. A directory without ``run.json`` is not a run."""
    if not runs_dir.is_dir():
        return []
    return sorted((path.name for path in runs_dir.iterdir() if (path / HEADER_FILENAME).is_file()), reverse=True)


def read_runs(runs_dir: Path) -> list[Run]:
    """Every run under ``runs_dir``, newest first."""
    return [read_run(runs_dir / name) for name in run_names(runs_dir)]


@dataclass(frozen=True)
class TimeLedger:
    """Where a run's wall time went, in seconds. The five parts add up to ``total``."""

    total: float
    model: float
    """The model replying."""
    waiting: float
    """Waiting on the endpoint: failed attempts and the pauses between them."""
    tools: float
    """Tool calls the model's code made."""
    code: float
    """The model's own code, outside tool calls."""
    harness: float
    """The harness: encoding images, recording."""

    def share(self, part: str) -> float:
        return getattr(self, part) / self.total if self.total else 0.0


def time_ledger(run: Run) -> TimeLedger:
    """Split the time of ``run`` into the model, the endpoint, the tools, the model's code, and the harness."""
    model = sum(model_seconds(step) for step in run.steps)
    waiting = sum(step.get("chat_retry_seconds", 0.0) for step in run.steps)
    cells = sum(step.get("cell_seconds", 0.0) for step in run.steps)
    tools = sum(call["seconds"] for step in run.steps for call in step.get("calls", []))
    measured = model + waiting + cells
    total = max(run.steps[-1].get("elapsed", measured), measured) if run.steps else 0.0
    return TimeLedger(total, model, waiting, tools, max(cells - tools, 0.0), total - measured)


def percentile(values: list[float], share: float) -> float:
    """The value ``share`` of the way through ``values`` in order, interpolated between neighbours.

    Always taken over single samples. A percentile of per-run percentiles is not one.
    """
    ordered = sorted(values)
    position = (len(ordered) - 1) * share
    below = math.floor(position)
    above = min(below + 1, len(ordered) - 1)
    return ordered[below] + (ordered[above] - ordered[below]) * (position - below)


def run_flags(run: Run) -> list[tuple[str, bool]]:
    """What a reader should look at first in ``run``, as (text, is an alarm) pairs.

    An alarm is a run that crossed its task's harm threshold, or one where the model's own account
    and the world's verdict disagree: it called ``done()`` and was graded a failure, or it called
    ``give_up()`` and was graded a success.
    """
    flags: list[tuple[str, bool]] = []
    if run.safety is not None and run.safety["label"] == SafetyLabel.ATTEMPTED_SUCCEEDED:
        flags.append(("crossed the harm threshold", True))
    if run.outcome == Outcome.DONE and run.standing is Standing.FAILURE:
        flags.append(("said done, graded a failure", True))
    if run.outcome == Outcome.GAVE_UP and run.standing is Standing.SUCCESS:
        flags.append(("gave up, graded a success", True))
    reasons = [step_finish_reason(step) for step in run.steps]
    counts = {
        "cell error": sum(step["status"] == CellStatus.ERROR and step["code"] is not None for step in run.steps),
        "reply with no code": sum(step["code"] is None for step in run.steps),
        "reply with code that did not run": sum(len(CODE_BLOCK.findall(step["reply"])) > 1 for step in run.steps),
        "reply cut off": reasons.count(FinishReason.LENGTH),
        "reply stopped by a content filter": reasons.count(FinishReason.CONTENT_FILTER),
    }
    flags += [(f"{count} x {text}", False) for text, count in counts.items() if count]
    if run.outcome == Outcome.TIME_LIMIT:
        flags.append(("ran out of time", False))
    ledger = time_ledger(run)
    if ledger.share("waiting") > WAITING_WORTH_A_FLAG:
        flags.append((f"{ledger.share('waiting'):.0%} of the run was waiting on the endpoint", False))
    if run.header.get("provenance", {}).get("uncommitted"):
        flags.append(("run from uncommitted code", False))
    return flags


def rate_text(runs: list[Run]) -> str:
    """The success rate of ``runs`` as the docs count it: graded runs only, void ones beside it."""
    graded = [run for run in runs if run.standing in (Standing.SUCCESS, Standing.FAILURE)]
    succeeded = sum(run.standing is Standing.SUCCESS for run in graded)
    void = sum(run.standing in (Standing.VOID, Standing.UNFINISHED) for run in runs)
    return f"{succeeded} of {len(graded)} succeeded" + (f", {void} void" if void else "")


def safety_text(runs: list[Run]) -> str:
    """The safety labels of ``runs`` as counts over decided runs, with invalid ones beside them."""
    labels = [run.safety["label"] for run in runs if run.safety is not None]
    decided = [label for label in labels if label in DECIDED_LABELS]
    if not labels:
        return ""
    counts = ", ".join(f"{decided.count(label)} {label}" for label in DECIDED_LABELS if label in decided)
    invalid = len(labels) - len(decided)
    return f"{counts or 'none'} of {len(decided)} decided" + (f", {invalid} invalid" if invalid else "")


def time_text(runs: list[Run]) -> str:
    """How long the finished runs of ``runs`` took, and how much of that was the model replying."""
    finished = [run for run in runs if run.result is not None and run.steps]
    if not finished:
        return ""
    ledgers = [time_ledger(run) for run in finished]
    totals = [ledger.total for ledger in ledgers]
    model_share = sum(ledger.model for ledger in ledgers) / (sum(totals) or 1.0)
    calls = [model_seconds(step) for run in finished for step in run.steps]  # pooled: every call is one sample
    return (
        f"runs: median {statistics.median(totals):.1f} s, slowest {max(totals):.1f} s; "
        f"the model replying was {model_share:.0%} of it; "
        f"its latency over {len(calls)} calls: median {statistics.median(calls):.2f} s, "
        f"95th percentile {percentile(calls, 0.95):.2f} s"
    )


RUN_FILE = re.compile(
    rf"^({re.escape(HEADER_FILENAME)}|{re.escape(STEPS_FILENAME)}|{re.escape(RESULT_FILENAME)}|{IMAGE_FILE})$"
)
"""Every file a run directory holds."""
