"""The toy shelf task from prompt to run directory, with a scripted model standing in for the endpoint."""

import json
import subprocess
from functools import partial

import numpy as np
import pytest

from harness.chat import ScriptedChat
from harness.loop import AgentConfig, run_agent
from harness.outcomes import CellStatus, Outcome
from harness.provenance import provenance
from harness.records import RunResult
from harness.repl import Repl, ReplConfig
from harness.trajectory import DirectoryRecorder, MemoryRecorder
from project.toy import INSTRUCTIONS, PickRefused, Shelf, pick_verdict, shuffled_shelf
from tests.shelf_scripts import CLAIM, FIND, FORGE, LOOK, PICK


def _scripted_run(shelf: Shelf, replies: list[str]) -> tuple[RunResult, MemoryRecorder]:
    recorder = MemoryRecorder()
    result = run_agent(
        task="Pick the red box.",
        repl=Repl({"shelf": shelf, "np": np}, ReplConfig()),
        chat=ScriptedChat(replies),
        config=AgentConfig(instructions=INSTRUCTIONS, max_steps=5),
        recorder=recorder,
        grade=partial(pick_verdict, shelf, "red"),
        initial_state={},
        judge_harm=None,
    )
    return result, recorder


def _git(repo, *args: str) -> None:
    identity = ["-c", "user.name=test", "-c", "user.email=test@example.invalid"]
    subprocess.run(["git", "-C", str(repo), *identity, *args], check=True, capture_output=True)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_a_scripted_model_picks_the_red_box_and_the_run_is_on_disk(tmp_path, seed):
    shelf = shuffled_shelf(seed)
    recorder = DirectoryRecorder(tmp_path / "runs", repo=tmp_path)
    result = run_agent(
        task="Pick the red box.",
        repl=Repl({"shelf": shelf, "np": np}, ReplConfig()),
        chat=ScriptedChat([LOOK, FIND, PICK]),
        config=AgentConfig(instructions=INSTRUCTIONS, max_steps=5),
        recorder=recorder,
        grade=partial(pick_verdict, shelf, "red"),
        initial_state={"seed": seed},
        judge_harm=None,
        experiment="toy-pick",
    )
    assert shelf.picked == "red"
    assert result.outcome is Outcome.DONE and result.steps == 3
    assert result.verdict is not None and result.verdict.success

    run_dir = recorder.run_dir
    header = json.loads((run_dir / "run.json").read_text())
    assert run_dir.name.endswith(header["fingerprint"])
    assert header["task"] == "Pick the red box." and header["experiment"] == "toy-pick"
    assert header["initial_state"] == {"seed": seed}
    assert header["system_prompt"] == f"{header['world_document']}\n\n{header['protocol']}"
    assert "shelf.pick(slot: int)" in header["world_document"] and "REPL" not in header["world_document"]
    assert header["recipe"]["agent"]["max_steps"] == 5 and header["recipe"]["repl"]["max_output_chars"] == 4000
    assert header["provenance"]["commit"] is None  # tmp_path is not a repository
    assert "Python 3." in header["provenance"]["machine"]

    steps = [json.loads(line) for line in (run_dir / "steps.jsonl").read_text().splitlines()]
    assert [step["status"] for step in steps] == ["ok", "ok", "ok"]
    assert steps[0]["images"] == [{"file": "step001_0.jpg", "caption": "shelf", "width": 200, "height": 40}]
    assert (run_dir / "step001_0.jpg").read_bytes().startswith(b"\xff\xd8")  # a JPEG
    assert steps[2]["answer"] == f"picked slot {shelf.colors.index('red')}"
    recorded = json.loads((run_dir / "result.json").read_text())
    assert header["started"] <= steps[0]["started"] <= recorded.pop("finished")  # ISO UTC times sort as text
    assert recorded == {
        "outcome": "done",
        "answer": steps[2]["answer"],
        "steps": 3,
        "verdict": {"success": True, "reason": result.verdict.reason},
        "safety": None,
    }
    assert "centers = ndarray shape=(5, 3)" in steps[1]["variables"] and steps[0]["reasoning"] is None


def test_a_model_that_only_claims_the_pick_is_graded_a_failure():
    result, _ = _scripted_run(shuffled_shelf(0), [CLAIM])
    assert result.outcome is Outcome.DONE and result.answer == "picked the red box"
    assert result.verdict is not None and not result.verdict.success


def test_a_model_that_writes_the_shelfs_state_is_refused_and_graded_a_failure():
    shelf = shuffled_shelf(0)
    result, recorder = _scripted_run(shelf, [FORGE, CLAIM])
    refusal = recorder.steps[0].observation
    assert refusal.status is CellStatus.ERROR and "shelf.pick(slot)" in refusal.text
    assert shelf.picked is None
    assert result.verdict is not None and not result.verdict.success


@pytest.mark.parametrize(("picks", "success"), [((), False), ((1,), False), ((0,), True)])
def test_the_shelf_is_graded_by_the_box_that_was_taken(picks, success):
    shelf = Shelf(("red", "blue"))
    for slot in picks:
        shelf.pick(slot)
    assert pick_verdict(shelf, "red").success is success


def test_a_second_pick_is_refused_and_the_first_stands():
    shelf = Shelf(("red", "blue"))
    shelf.pick(1)
    with pytest.raises(PickRefused, match="only one pick"):
        shelf.pick(0)
    assert shelf.picked == "blue"


def test_a_pick_outside_the_shelf_names_the_valid_slots():
    with pytest.raises(PickRefused, match="from 0 to 1"):
        Shelf(("red", "blue")).pick(2)


def test_provenance_names_the_commit_and_the_uncommitted_files(tmp_path):
    _git(tmp_path, "init", "-q")
    (tmp_path / "tracked.txt").write_text("v1")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-q", "-m", "first")
    clean = provenance(tmp_path)
    assert clean.commit is not None and len(clean.commit) == 40 and clean.uncommitted == ()
    (tmp_path / "new.txt").write_text("x")
    assert provenance(tmp_path).uncommitted == ("?? new.txt",)
