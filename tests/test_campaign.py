"""Campaigns on the toy shelf: a program carried across rounds, a judge that keeps or reverts it, and held out."""

import json
from functools import partial

import numpy as np
import pytest

from harness.campaign import (
    Briefing,
    CampaignConfig,
    CampaignEnding,
    DirectoryCampaignRecorder,
    Judge,
    JudgeAgreement,
    MemoryCampaignRecorder,
    Practice,
    Round,
    SelfJudge,
    judge_agreement,
    run_campaign,
)
from harness.chat import ChatError, ScriptedChat
from harness.library import Library
from harness.loop import AgentConfig, run_agent
from harness.repl import Repl, ReplConfig
from harness.trajectory import MemoryRecorder
from project.toy import BOX_SIDE, INSTRUCTIONS, pick_verdict, shuffled_shelf
from project.toy_revision import POLICY, ProgramLoop, SealedProgramJudge
from tests.shelf_scripts import CLAIM

JUDGE_SEEDS = (100, 101, 102)
HELD_OUT_SEEDS = (200, 201, 202)
REFERENCE = (
    "def policy(shelf):\n"
    "    image = shelf.photo()\n"
    f"    centers = image[{BOX_SIDE // 2}, {BOX_SIDE // 2}::{BOX_SIDE}]\n"
    "    shelf.pick(int(np.argmax((centers == (0, 0, 255)).all(axis=1))))\n"
)
READS_STATE = "def policy(shelf):\n    shelf.pick(shelf.colors.index('red'))\n"
"""Passes wherever the shelf exposes its colors, as a simulator exposes its assets."""
BROKEN = "def policy(shelf):\n    shelf.pick(99)\n"

SEALED = SealedProgramJudge("red", JUDGE_SEEDS)


def store(program: str) -> str:
    return f"Store it.\n```repl\nlibrary.write({POLICY!r}, {program!r})\ndone('stored a policy')\n```"


def campaign(
    replies: list[str],
    rounds: int,
    briefing: Briefing = Briefing.JUDGE,
    judge: Judge = SEALED,
    recorder: MemoryCampaignRecorder | DirectoryCampaignRecorder | None = None,
) -> tuple[list[Round], Library, ScriptedChat]:
    chat, library = ScriptedChat(replies), Library()

    def practice(index: int, task: str) -> Practice:
        shelf = shuffled_shelf(index)
        result = run_agent(
            task=task,
            repl=Repl({"shelf": shelf, "library": library, "np": np}, ReplConfig()),
            chat=chat,
            config=AgentConfig(instructions=INSTRUCTIONS, max_steps=3),
            recorder=MemoryRecorder(),
            grade=partial(pick_verdict, shelf, "red"),
            initial_state={"seed": index},
            judge_harm=None,
        )
        return Practice(result, None)

    records = run_campaign(
        loop=ProgramLoop("red", HELD_OUT_SEEDS),
        judge=judge,
        config=CampaignConfig(rounds, briefing),
        library=library,
        practice=practice,
        recorder=recorder or MemoryCampaignRecorder(),
    )
    return records, library, chat


def test_a_reference_program_is_kept_and_passes_held_out():
    records, library, _ = campaign([store(REFERENCE)], rounds=1)
    (record,) = records
    assert record.kept and record.held_out is not None and record.held_out.success
    assert library.read(POLICY) == REFERENCE
    assert judge_agreement(records) == JudgeAgreement(1, 0, 0, 0, 0)


def test_a_model_that_only_claims_done_is_rejected_and_fails_held_out():
    records, library, _ = campaign([CLAIM], rounds=1)
    (record,) = records
    assert record.acceptance is not None and not record.acceptance.accepted
    assert record.held_out is not None and not record.held_out.success
    assert library.names() == []
    assert judge_agreement(records) == JudgeAgreement(0, 0, 0, 1, 0)


def test_a_program_that_reads_the_shelfs_state_passes_the_judge_and_fails_the_world():
    records, _, _ = campaign([store(READS_STATE)], rounds=1)
    (record,) = records
    assert record.kept
    assert record.held_out is not None and "AttributeError" in record.held_out.reason
    assert judge_agreement(records).judge_only == 1


@pytest.mark.parametrize(
    ("judge", "agreement"), [(SEALED, JudgeAgreement(0, 0, 0, 1, 0)), (SelfJudge(), JudgeAgreement(0, 1, 0, 0, 0))]
)
def test_the_same_loop_and_model_under_another_judge_keeps_what_the_sealed_judge_rejected(judge, agreement):
    records, _, _ = campaign([CLAIM], rounds=1, judge=judge)
    assert judge_agreement(records) == agreement


def test_a_rejected_round_leaves_the_library_as_the_last_kept_round_left_it():
    records, library, _ = campaign([store(REFERENCE), store(BROKEN)], rounds=2)
    assert [record.kept for record in records] == [True, False]
    assert records[1].written == {POLICY: BROKEN}
    assert library.read(POLICY) == REFERENCE


@pytest.mark.parametrize(("briefing", "told_held_out"), [(Briefing.JUDGE, False), (Briefing.JUDGE_AND_HELD_OUT, True)])
def test_the_next_round_hears_the_judge_and_hears_held_out_only_when_briefed_to(briefing, told_held_out):
    records, _, chat = campaign([store(BROKEN), store(REFERENCE)], rounds=2, briefing=briefing)
    second_task = chat.prompts[1][1]["content"]
    assert second_task == f"Task: {records[1].task}"
    assert "put back as it was before: the program picked the red box on 0 of 3 shelves" in second_task
    assert ("conditions the judge did not use" in second_task) is told_held_out


def test_a_campaign_is_written_to_its_directory(tmp_path):
    recorder = DirectoryCampaignRecorder(tmp_path / "campaigns", repo=tmp_path)
    campaign([store(REFERENCE), store(BROKEN)], rounds=2, recorder=recorder)
    header = json.loads((recorder.campaign_dir / "campaign.json").read_text())
    rounds = [json.loads(line) for line in (recorder.campaign_dir / "rounds.jsonl").read_text().splitlines()]
    end = json.loads((recorder.campaign_dir / "end.json").read_text())
    assert header["loop"] == {"kind": "program", "color": "red", "held_out_seeds": list(HELD_OUT_SEEDS)}
    assert header["judge"] == {"kind": "sealed", "color": "red", "seeds": list(JUDGE_SEEDS)}
    assert header["config"] == {"rounds": 2, "briefing": "judge"}
    assert [record["kept"] for record in rounds] == [True, False]
    assert rounds[0]["written"] == {POLICY: REFERENCE}
    assert end["ending"] == "completed"


def test_a_crash_mid_round_restores_the_library_and_ends_the_campaign_crashed():
    recorder = MemoryCampaignRecorder()
    write_only = f"Store it.\n```repl\nlibrary.write({POLICY!r}, {BROKEN!r})\n```"
    with pytest.raises(ChatError):
        campaign([store(REFERENCE), write_only], rounds=2, recorder=recorder)
    assert recorder.endings == [CampaignEnding.CRASHED]
    assert len(recorder.rounds) == 1
