"""The toy shelf task from prompt to run directory, with a scripted model standing in for the endpoint."""

import json
import subprocess

import numpy as np
import pytest

from harness.chat import ScriptedChat
from harness.loop import AgentConfig, run_agent
from harness.provenance import provenance
from harness.records import Outcome
from harness.repl import Repl, ReplConfig
from harness.trajectory import DirectoryRecorder
from project.toy import BOX_SIDE, INSTRUCTIONS, PickRefused, Shelf, shuffled_shelf

LOOK = "Look first.\n```repl\nimg = shelf.photo()\nshow(img, 'shelf')\nimg.shape\n```"
FIND = (
    "Red is (0, 0, 255) in BGR.\n```repl\n"
    f"centers = img[{BOX_SIDE // 2}, {BOX_SIDE // 2}::{BOX_SIDE}]\n"
    "slot = int(np.argmax((centers == (0, 0, 255)).all(axis=1)))\nslot\n```"
)
PICK = "Pick it.\n```repl\nshelf.pick(slot)\ndone(f'picked slot {slot}')\n```"


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
        experiment="toy-pick",
    )
    assert shelf.picked == "red"
    assert result.outcome is Outcome.DONE and result.steps == 3

    run_dir = recorder.run_dir
    header = json.loads((run_dir / "run.json").read_text())
    assert run_dir.name.endswith(header["fingerprint"])
    assert header["task"] == "Pick the red box." and header["experiment"] == "toy-pick"
    assert header["recipe"]["agent"]["max_steps"] == 5 and header["recipe"]["repl"]["max_output_chars"] == 4000
    assert header["provenance"]["commit"] is None  # tmp_path is not a repository

    steps = [json.loads(line) for line in (run_dir / "steps.jsonl").read_text().splitlines()]
    assert [step["status"] for step in steps] == ["ok", "ok", "ok"]
    assert steps[0]["images"] == [{"file": "step001_0.jpg", "caption": "shelf", "width": 200, "height": 40}]
    assert (run_dir / "step001_0.jpg").read_bytes().startswith(b"\xff\xd8")  # a JPEG
    assert steps[2]["answer"] == f"picked slot {shelf.colors.index('red')}"
    assert json.loads((run_dir / "result.json").read_text()) == {
        "outcome": "done",
        "answer": steps[2]["answer"],
        "steps": 3,
    }


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
