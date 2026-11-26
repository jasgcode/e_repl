"""Recorders: where a run's header, steps, and result go as the loop produces them.

A run directory is a trajectory: the system prompt, every reply, the code that ran, what came back,
and the JPEGs the model was shown. Steps are appended as they happen, so a run that is interrupted
or crashes keeps everything up to that point.

    runs/20261117-142501-3f9a1c2e/     start time, then the recipe's fingerprint
      run.json                         task, initial state, prompt, recipe, experiment, provenance, start time
      steps.jsonl                      one line per step: all the model wrote and thought, all it was shown
      step003_0.jpg                    the images of step 3
      result.json                      outcome, answer, step count, the world's verdict, finish time

A record is for reading later by people who were not there, so a step keeps what cannot be worked
out afterwards: the model's reasoning, the endpoint's whole response, and the times.
"""

import dataclasses
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from harness.provenance import Provenance, provenance, utc_timestamp
from harness.records import AgentPrompt, RunHeader, RunResult, Step
from harness.runs import HEADER_FILENAME, RESULT_FILENAME, STEPS_FILENAME, image_filename

RUN_TIME_FORMAT = "%Y%m%d-%H%M%S"


class Recorder(Protocol):
    def start(self, header: RunHeader) -> None: ...

    def step(self, step: Step) -> None: ...

    def finish(self, result: RunResult) -> None: ...


@dataclass
class MemoryRecorder:
    """Keeps a run in memory: for tests, and for callers that want the steps back."""

    headers: list[RunHeader] = field(default_factory=list)
    steps: list[Step] = field(default_factory=list)
    results: list[RunResult] = field(default_factory=list)

    def start(self, header: RunHeader) -> None:
        self.headers.append(header)

    def step(self, step: Step) -> None:
        self.steps.append(step)

    def finish(self, result: RunResult) -> None:
        self.results.append(result)


def _prompt_record(prompt: AgentPrompt) -> dict[str, str]:
    return {
        "system_prompt": prompt.system_prompt,
        "world_document": prompt.world_document,
        "protocol": prompt.protocol,
    }


def header_record(header: RunHeader, source: Provenance, started: float) -> dict[str, Any]:
    """``run.json``. One agent's prompt is at the top level; a team's are under ``team``, one per agent."""
    single = len(header.prompts) == 1
    unset = dict.fromkeys(_prompt_record(header.prompts[0]))
    return {
        "task": header.task,
        "started": utc_timestamp(started),
        "initial_state": header.initial_state,
        "experiment": header.experiment,
        "fingerprint": header.fingerprint,
        "recipe": header.recipe,
        **(_prompt_record(header.prompts[0]) if single else unset),
        "team": None if single else [{"agent": prompt.agent, **_prompt_record(prompt)} for prompt in header.prompts],
        "provenance": dataclasses.asdict(source),
    }


def step_record(step: Step) -> dict[str, Any]:
    """One ``steps.jsonl`` line; images are named by ``image_filename``, not embedded."""
    observation = step.observation
    return {
        "step": step.index,
        "agent": step.agent,
        "started": utc_timestamp(step.started),
        "heard": list(step.heard),
        "told": list(step.told),
        "reasoning": step.reply.reasoning,
        "reply": step.reply.text,
        "code": step.code,
        "status": observation.status,
        "output": observation.text,
        "variables": step.variables,
        "images": [
            {
                "file": image_filename(step.index, i),
                "caption": image.caption,
                "width": image.width,
                "height": image.height,
            }
            for i, image in enumerate(observation.images)
        ],
        "ending": observation.ending,
        "answer": observation.answer,
        "elapsed": round(step.elapsed, 3),
        "time_left": None if step.time_left is None else round(step.time_left, 3),
        "chat_seconds": round(step.reply.seconds, 3),
        "chat_request_seconds": round(step.reply.request_seconds, 3),
        "chat_retry_seconds": round(step.reply.seconds - step.reply.request_seconds, 3),
        "chat_attempts": step.reply.attempts,
        "cell_seconds": round(step.cell_seconds, 3),
        "calls": [
            {
                "name": call.name,
                "at": round(call.at, 3),
                "seconds": round(call.seconds, 3),
                "raised": call.raised,
                "facts": call.facts,
            }
            for call in observation.calls
        ],
        "response": step.reply.response,
    }


def _write_json(path: Path, record: dict[str, Any]) -> None:
    path.write_text(json.dumps(record, indent=2) + "\n")


class DirectoryRecorder:
    """Writes each run to its own directory under ``runs_dir``.

    ``repo`` is the checkout whose commit and uncommitted files the run's provenance records.
    """

    run_dir: Path
    """Set by ``start``."""

    def __init__(self, runs_dir: Path, repo: Path):
        self.runs_dir = runs_dir
        self.repo = repo

    def start(self, header: RunHeader) -> None:
        self.run_dir = self.runs_dir / f"{time.strftime(RUN_TIME_FORMAT)}-{header.fingerprint}"
        self.run_dir.mkdir(parents=True)  # two runs of one recipe in one second: fail, never merge
        _write_json(self.run_dir / HEADER_FILENAME, header_record(header, provenance(self.repo), time.time()))

    def step(self, step: Step) -> None:
        for i, image in enumerate(step.observation.images):
            (self.run_dir / image_filename(step.index, i)).write_bytes(image.jpeg)
        with open(self.run_dir / STEPS_FILENAME, "a") as f:
            f.write(json.dumps(step_record(step)) + "\n")

    def finish(self, result: RunResult) -> None:
        record = {**dataclasses.asdict(result), "finished": utc_timestamp(time.time())}
        _write_json(self.run_dir / RESULT_FILENAME, record)
