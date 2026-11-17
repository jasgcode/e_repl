"""The agent loop with a scripted model: what the model is sent, and how each run ends."""

import numpy as np
import pytest

from harness.chat import ChatError, ScriptedChat
from harness.loop import NO_CODE_BLOCK, AgentConfig, run_agent
from harness.records import Outcome, RunResult
from harness.repl import CellStatus, Repl, ReplConfig
from harness.trajectory import MemoryRecorder

INSTRUCTIONS = "You sort parcels."


def _run(replies, tools=None, config: AgentConfig | None = None, task: str = "a task"):
    chat = ScriptedChat(replies)
    recorder = MemoryRecorder()
    config = config or AgentConfig(instructions=INSTRUCTIONS, max_steps=10)
    result = run_agent(task, Repl(tools or {}, ReplConfig()), chat, config, recorder)
    return result, chat, recorder


def _cell(code: str) -> str:
    return f"Thinking.\n```repl\n{code}\n```"


def _image_count(prompt) -> int:
    return sum(part["type"] == "image_url" for m in prompt if isinstance(m["content"], list) for part in m["content"])


def test_the_run_ends_with_the_answer_passed_to_done():
    result, chat, recorder = _run([_cell("x = 2"), _cell("done(f'x is {x}')")])
    assert result == RunResult(Outcome.DONE, "x is 2", steps=2)
    assert recorder.results == [result]
    assert [step.code for step in recorder.steps] == ["x = 2\n", "done(f'x is {x}')\n"]
    assert len(chat.prompts) == 2


def test_the_first_prompt_holds_the_instructions_the_tools_and_the_task():
    def weigh(parcel: str) -> float:
        """Weight of a parcel in kilograms."""
        return 1.0

    _, chat, recorder = _run([_cell("done()")], tools={"weigh": weigh}, task="sort the red parcel")
    system, user = chat.prompts[0]
    assert system["content"].startswith(INSTRUCTIONS)
    assert "- weigh(parcel: str) -> float: Weight of a parcel in kilograms." in system["content"]
    assert user["content"] == "Task: sort the red parcel"
    assert recorder.headers[0].system_prompt == system["content"]


def test_an_observation_carries_the_output_and_the_variables():
    _, chat, _ = _run([_cell("total = 40 + 2\nprint('hi')"), _cell("done()")])
    observation = chat.prompts[1][-1]["content"][0]["text"]
    assert observation == "[step 1 ok]\nhi\n\n[variables]\ntotal = 42"


def test_a_reply_without_a_code_block_is_answered_with_the_format():
    result, chat, recorder = _run(["I forgot the block.", _cell("done('ok')")])
    assert result.outcome is Outcome.DONE and result.steps == 2
    assert NO_CODE_BLOCK in chat.prompts[1][-1]["content"][0]["text"]
    assert recorder.steps[0].code is None and recorder.steps[0].observation.status is CellStatus.ERROR


def test_only_the_latest_images_stay_in_the_context():
    show = "show(np.zeros((8, 8), np.uint8))"
    config = AgentConfig(instructions=INSTRUCTIONS, max_steps=10, keep_images=1)
    _, chat, recorder = _run([_cell(show), _cell(show), _cell(show), _cell("done()")], {"np": np}, config)
    assert [_image_count(prompt) for prompt in chat.prompts] == [0, 1, 1, 1]
    assert "[older image removed]" in str(chat.prompts[-1])
    assert [len(step.observation.images) for step in recorder.steps] == [1, 1, 1, 0]  # the record keeps them all


def test_a_run_that_never_calls_done_stops_at_the_step_limit():
    config = AgentConfig(instructions=INSTRUCTIONS, max_steps=3)
    result, chat, _ = _run([_cell("1")] * 5, config=config)
    assert result == RunResult(Outcome.STEP_LIMIT, None, steps=3)
    assert len(chat.prompts) == 3


def test_a_failing_chat_model_is_recorded_as_a_crash_and_raises():
    chat, recorder = ScriptedChat([_cell("1")]), MemoryRecorder()
    config = AgentConfig(instructions=INSTRUCTIONS, max_steps=5)
    with pytest.raises(ChatError):
        run_agent("a task", Repl({}, ReplConfig()), chat, config, recorder)
    assert recorder.results == [RunResult(Outcome.CRASHED, None, steps=1)]


def test_ctrl_c_during_a_cell_is_recorded_as_interrupted_and_raises():
    def stop() -> None:
        raise KeyboardInterrupt

    chat, recorder = ScriptedChat([_cell("1"), _cell("stop()")]), MemoryRecorder()
    config = AgentConfig(instructions=INSTRUCTIONS, max_steps=5)
    with pytest.raises(KeyboardInterrupt):
        run_agent("a task", Repl({"stop": stop}, ReplConfig()), chat, config, recorder)
    assert recorder.results == [RunResult(Outcome.INTERRUPTED, None, steps=1)]


def test_the_fingerprint_follows_the_recipe_and_not_the_task():
    def fingerprint(task: str, max_steps: int) -> str:
        config = AgentConfig(instructions=INSTRUCTIONS, max_steps=max_steps)
        _, _, recorder = _run([_cell("done()")], config=config, task=task)
        return recorder.headers[0].fingerprint

    assert fingerprint("pick red", 10) == fingerprint("pick blue", 10)
    assert fingerprint("pick red", 10) != fingerprint("pick red", 11)
