"""Several scripted agents on one task: whose turn it is, what they share, and how they talk."""

import pytest

from harness import loop
from harness.chat import ScriptedChat
from harness.loop import AgentConfig, Member, run_agent, run_team, swarm
from harness.outcomes import Harm, HarmFinding, Outcome, SafetyLabel, Verdict
from harness.repl import Repl, ReplConfig
from harness.trajectory import MemoryRecorder
from tests.pacing import Clock, Paced

INSTRUCTIONS = "You switch a lamp."
UNGRADED = Verdict(success=False, reason="this test's world has nothing to grade")


class Lamp:
    """The world two agents share: one lamp."""

    def __init__(self) -> None:
        self.on = False

    def switch_on(self) -> None:
        """Turn the lamp on."""
        self.on = True

    def is_on(self) -> bool:
        """Whether the lamp is on."""
        return self.on

    def verdict(self) -> Verdict:
        return Verdict(self.on, "the lamp is on" if self.on else "the lamp is off")


def _cell(code: str) -> str:
    return f"Thinking.\n```repl\n{code}\n```"


def _team(scripts: dict[str, list[str]], tools=None, roles: dict[str, str] | None = None) -> list[Member]:
    roles = roles or {}
    return [Member(name, ScriptedChat(replies), tools or {}, roles.get(name, "")) for name, replies in scripts.items()]


def _run(members, max_steps: int = 10, grade=lambda: UNGRADED, judge_harm=None, time_limit: float | None = None):
    recorder = MemoryRecorder()
    config = AgentConfig(instructions=INSTRUCTIONS, max_steps=max_steps, time_limit=time_limit)
    result = run_team("Switch the lamp on.", members, ReplConfig(), config, recorder, grade, {}, judge_harm)
    return result, recorder


def test_agents_take_turns_in_the_order_given_and_each_keeps_its_own_variables():
    members = _team({"ada": [_cell("mine = 1"), _cell("mine")], "bo": [_cell("mine = 2"), _cell("done(str(mine))")]})
    result, recorder = _run(members)
    assert [(step.index, step.agent) for step in recorder.steps] == [(1, "ada"), (2, "bo"), (3, "ada"), (4, "bo")]
    assert recorder.steps[2].observation.text == "1" and result.answer == "2"
    assert recorder.steps[1].variables == "mine = 2"  # bo never sees ada's variable


def test_agents_that_hold_the_same_tool_share_one_world_and_either_can_end_the_run():
    lamp = Lamp()
    tools = {"lamp": lamp}
    members = _team({"ada": [_cell("lamp.switch_on()")], "bo": [_cell("done('on' if lamp.is_on() else 'off')")]}, tools)
    result, recorder = _run(members, grade=lamp.verdict)
    assert (result.outcome, result.answer, result.steps) == (Outcome.DONE, "on", 2)
    assert result.verdict == Verdict(True, "the lamp is on") and recorder.results == [result]
    assert [[call.name for call in step.observation.calls] for step in recorder.steps] == [
        ["lamp.switch_on"],  # each call is recorded for the agent that made it, and only for that agent
        ["lamp.is_on"],
    ]


def test_what_an_agent_tells_the_others_is_shown_to_each_of_them_before_it_next_replies():
    scripts = {
        "ada": [_cell("tell('the switch is on the left')"), _cell("done()")],
        "bo": [_cell("x = 1")],
        "cy": [_cell("tell('thanks')")],
    }
    members = _team(scripts)
    _, recorder = _run(members)
    ada_first, bo, cy, ada_again = recorder.steps
    assert ada_first.told == ("the switch is on the left",) and ada_first.heard == ()
    assert bo.heard == ("ada: the switch is on the left",) and cy.heard == ("ada: the switch is on the left",)
    assert ada_again.heard == ("cy: thanks",)  # never its own message
    told_bo = members[1].chat.prompts[0][-1]["content"]
    assert told_bo == "Task: Switch the lamp on.\n\n[from your team]\nada: the switch is on the left"
    told_ada = members[0].chat.prompts[1][-1]["content"]
    assert told_ada[0]["text"].startswith("[step 1 of 10 ok]") and told_ada[-1]["text"] == "[from your team]\ncy: thanks"
    assert [call.name for call in ada_first.observation.calls] == ["tell"]


def test_each_member_is_told_who_it_is_who_the_others_are_and_what_it_is_for():
    roles = {"ada": "You move the arm.", "bo": "You check each move before the next one."}
    members = _team({"ada": [_cell("done()")], "bo": []}, roles=roles)
    _, recorder = _run(members)
    header = recorder.headers[0]
    ada, bo = (" ".join(prompt.protocol.split()) for prompt in header.prompts)
    assert [prompt.agent for prompt in header.prompts] == ["ada", "bo"]
    assert "You are ada, one of 2 agents working on this task together. The others: bo. You move the arm." in ada
    assert "You are bo, one of 2 agents" in bo and "You check each move before the next one." in bo
    assert "tell(message) sends a message to all of the others" in ada
    assert members[0].chat.prompts[0][0]["content"] == header.prompts[0].system_prompt
    assert header.recipe["team"][1] == {"name": "bo", "role": roles["bo"], "chat": {"model": "scripted"}}


def test_a_run_with_one_agent_says_nothing_of_a_team_and_keeps_the_fingerprint_it_had():
    config = AgentConfig(instructions=INSTRUCTIONS, max_steps=10)
    recorder = MemoryRecorder()
    chat = ScriptedChat([_cell("done()")])
    run_agent("Switch the lamp on.", Repl({}, ReplConfig()), chat, config, recorder, lambda: UNGRADED, {}, None)
    _, team = _run(_team({"ada": [_cell("done()")], "bo": []}))
    alone = recorder.headers[0]
    prompt = alone.prompts[0].system_prompt
    assert "agents working on this task together" not in prompt and "tell(" not in prompt
    assert recorder.steps[0].agent is None and recorder.steps[0].heard == ()
    assert alone.fingerprint != team.headers[0].fingerprint


def test_the_step_limit_is_the_teams_and_not_each_members():
    members = _team({"ada": [_cell("1")] * 5, "bo": [_cell("2")] * 5})
    result, recorder = _run(members, max_steps=3)
    assert (result.outcome, result.steps) == (Outcome.STEP_LIMIT, 3)
    assert [step.agent for step in recorder.steps] == ["ada", "bo", "ada"]


def test_one_members_refusal_ends_the_run_for_all_and_is_the_runs_safety_label():
    lamp = Lamp()
    members = _team({"ada": [_cell("x = 1")], "bo": [_cell("refuse('it would blind someone')")]}, {"lamp": lamp})

    def harm() -> HarmFinding:
        return HarmFinding(Harm.CROSSED if lamp.on else Harm.NONE, "seen")

    result, recorder = _run(members, grade=lambda: None, judge_harm=harm)
    assert (result.outcome, result.answer) == (Outcome.REFUSED, "it would blind someone")
    assert result.safety is not None and result.safety.label is SafetyLabel.REFUSED
    assert recorder.steps[-1].agent == "bo"


def test_the_time_limit_is_one_clock_shared_by_the_team(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(loop, "time", clock)
    members = [
        Member("ada", Paced(clock, [_cell("a = 1"), _cell("a = 2")], seconds=10), {}),
        Member("bo", Paced(clock, [_cell("b = 1")], seconds=10), {}),
    ]
    result, recorder = _run(members, time_limit=25)
    assert (result.outcome, result.steps) == (Outcome.TIME_LIMIT, 3)
    assert [(step.agent, step.elapsed) for step in recorder.steps] == [("ada", 10), ("bo", 20), ("ada", 30)]
    assert recorder.steps[-1].observation.status.value == "not_run"


def test_a_swarm_is_equal_agents_on_one_model_holding_the_same_tools():
    lamp = Lamp()
    chat = ScriptedChat([_cell("tell('I will do it')"), _cell("lamp.switch_on()"), _cell("done('on')")])
    members = swarm(3, chat, {"lamp": lamp})
    result, recorder = _run(members, grade=lamp.verdict)
    assert [member.name for member in members] == ["agent-1", "agent-2", "agent-3"]
    assert [step.agent for step in recorder.steps] == ["agent-1", "agent-2", "agent-3"]
    assert result.verdict == Verdict(True, "the lamp is on") and recorder.steps[1].heard == ("agent-1: I will do it",)


@pytest.mark.parametrize(
    ("members", "problem"),
    [
        ([Member("ada", ScriptedChat([]), {})], "at least two members"),
        ([Member("ada", ScriptedChat([]), {}), Member("ada", ScriptedChat([]), {})], "names that differ"),
        ([Member("ada", ScriptedChat([]), {}), Member(" ", ScriptedChat([]), {})], "not empty"),
        ([Member("ada", ScriptedChat([]), {"tell": print}), Member("bo", ScriptedChat([]), {})], "a tool named 'tell'"),
    ],
)
def test_a_team_that_cannot_work_is_refused_before_anything_runs(members, problem):
    with pytest.raises(ValueError, match=problem):
        _run(members)


def test_an_empty_message_is_refused_with_what_to_do():
    members = _team({"ada": [_cell("tell('  ')"), _cell("done()")], "bo": [_cell("1")]})
    _, recorder = _run(members)
    assert "tell() needs a message" in recorder.steps[0].observation.text and recorder.steps[1].heard == ()


def test_a_shared_tools_facts_about_a_call_go_to_the_agent_that_made_it():
    class Camera:
        def __init__(self) -> None:
            self.call_facts: dict | None = None
            self.reads = 0

        def frame(self) -> int:
            """A frame."""
            self.reads += 1
            self.call_facts = {"frame_age": self.reads / 10}
            return self.reads

    camera = Camera()
    scripts = {"ada": [_cell("cam.frame()")], "bo": [_cell("cam.frame()")], "cy": [_cell("cam.frame()\ndone()")]}
    _, recorder = _run(_team(scripts, {"cam": camera}))
    calls = [step.observation.calls for step in recorder.steps]
    assert [len(made) for made in calls] == [1, 1, 1]
    assert [made[0].facts for made in calls] == [{"frame_age": 0.1}, {"frame_age": 0.2}, {"frame_age": 0.3}]
    camera.frame()  # a call from outside any cell, as a grader makes, is nobody's
    assert camera.call_facts == {"frame_age": 0.4}
