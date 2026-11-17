"""Recorders: where a run's header, steps, and result go as the loop produces them.

A run directory is a trajectory: the system prompt, every reply, the code that ran, what came back,
and the JPEGs the model was shown. Steps are appended as they happen, so a run that is interrupted
or crashes keeps everything up to that point.

    runs/20261117-142501-3f9a1c2e/     start time, then the recipe's fingerprint
      run.json                         task, system prompt, recipe, experiment, provenance
      steps.jsonl                      one line per step
      step003_0.jpg                    the images of step 3
      result.json                      outcome, answer, step count
"""

import dataclasses
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from harness.provenance import Provenance, provenance
from harness.records import RunHeader, RunResult, Step

HEADER_FILENAME = "run.json"
STEPS_FILENAME = "steps.jsonl"
RESULT_FILENAME = "result.json"
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


def image_filename(step: int, image: int) -> str:
    return f"step{step:03d}_{image}.jpg"


def header_record(header: RunHeader, source: Provenance) -> dict[str, Any]:
    return {
        "task": header.task,
        "experiment": header.experiment,
        "fingerprint": header.fingerprint,
        "recipe": header.recipe,
        "system_prompt": header.system_prompt,
        "provenance": dataclasses.asdict(source),
    }


def step_record(step: Step) -> dict[str, Any]:
    """One ``steps.jsonl`` line; images are named by ``image_filename``, not embedded."""
    observation = step.observation
    return {
        "step": step.index,
        "reply": step.reply,
        "code": step.code,
        "status": observation.status,
        "output": observation.text,
        "images": [
            {
                "file": image_filename(step.index, i),
                "caption": image.caption,
                "width": image.width,
                "height": image.height,
            }
            for i, image in enumerate(observation.images)
        ],
        "answer": observation.answer,
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
        _write_json(self.run_dir / HEADER_FILENAME, header_record(header, provenance(self.repo)))

    def step(self, step: Step) -> None:
        for i, image in enumerate(step.observation.images):
            (self.run_dir / image_filename(step.index, i)).write_bytes(image.jpeg)
        with open(self.run_dir / STEPS_FILENAME, "a") as f:
            f.write(json.dumps(step_record(step)) + "\n")

    def finish(self, result: RunResult) -> None:
        _write_json(self.run_dir / RESULT_FILENAME, dataclasses.asdict(result))
