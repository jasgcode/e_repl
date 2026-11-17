"""The agent loop: the model writes one cell, the REPL runs it, the observation goes back.

Every run is handed to a ``Recorder`` as it happens, so a successful run doubles as a trajectory to
filter and fine-tune on, and an interrupted one is kept as the failure it was.
"""

import dataclasses
import logging
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from harness.chat import ChatModel, Message, image_part
from harness.records import Outcome, RunHeader, RunResult, Step
from harness.repl import CellStatus, Observation, Repl, describe_tools
from harness.trajectory import Recorder

logger = logging.getLogger(__name__)

PROTOCOL = """Each turn, write one or two sentences of reasoning, then exactly one code block:
```repl
# your code
```
The code runs right away and variables persist between turns. You then see what it printed
(truncated to {max_output_chars} characters), the value of a final bare expression, any images
passed to show(), and a one-line summary of each variable you defined.

Work in small steps and check each result before building on it. Keep large data (images, arrays,
long lists) in variables and print short summaries. When a call fails, read the error, adjust,
and retry. Call done("summary") when the task is complete or clearly impossible.

Preloaded names:
- show(image, caption=""): attach a uint8 image (HxW gray or HxWx3 BGR) to your next observation
- done(answer=""): finish the task
{tools}
"""

CODE_BLOCK = re.compile(r"```(?:repl|python|py)?[ \t]*\n(.*?)```", re.DOTALL)
NO_CODE_BLOCK = "No ```repl code block found. Reply with exactly one, or call done() inside one."
OLD_IMAGE_PLACEHOLDER = "[older image removed]"


@dataclass(frozen=True)
class AgentConfig:
    """How the loop runs. Part of a run's recipe, with the REPL's config and the chat model's."""

    instructions: str
    """The project's part of the system prompt: who the model is, the world, the units, the rules."""
    max_steps: int
    keep_images: int = 2
    """Images stay in the context for this many of the latest observations that carry any."""


def system_prompt(instructions: str, tools: Mapping[str, Any], max_output_chars: int) -> str:
    protocol = PROTOCOL.format(max_output_chars=max_output_chars, tools=describe_tools(tools))
    return f"{instructions.strip()}\n\n{protocol}"


def extract_code(reply: str) -> str | None:
    """The cell to run from a model's reply, or ``None`` when it holds none.

    TODO(johnny): decide what a reply with several code blocks means. This baseline runs the first
    and silently drops the rest. See README.md, "The one policy left open".
    """
    match = CODE_BLOCK.search(reply)
    return match.group(1) if match else None


def observation_message(observation: Observation, variables: str, step: int) -> Message:
    text = f"[step {step} {observation.status}]\n{observation.text}\n\n[variables]\n{variables}"
    content: list[Message] = [{"type": "text", "text": text}]
    for image in observation.images:
        if image.caption:
            content.append({"type": "text", "text": f"[image: {image.caption}]"})
        content.append(image_part(image))
    return {"role": "user", "content": content}


def elide_old_images(messages: list[Message], keep: int) -> None:
    """Replace the images in all but the last ``keep`` image-bearing messages with a placeholder."""
    seen = 0
    for message in reversed(messages):
        content = message["content"]
        if not isinstance(content, list) or not any(part["type"] == "image_url" for part in content):
            continue
        seen += 1
        if seen > keep:
            message["content"] = [
                {"type": "text", "text": OLD_IMAGE_PLACEHOLDER} if part["type"] == "image_url" else part
                for part in content
            ]


def run_agent(
    task: str,
    repl: Repl,
    chat: ChatModel,
    config: AgentConfig,
    recorder: Recorder,
    experiment: str | None = None,
) -> RunResult:
    """Run until the model calls ``done()`` or ``config.max_steps`` cells have run.

    An exception from outside the model's code (the chat endpoint, Ctrl-C) is recorded as the run's
    outcome and then propagates.
    """
    system = system_prompt(config.instructions, repl.tools, repl.config.max_output_chars)
    recipe = {
        "agent": dataclasses.asdict(config),
        "repl": dataclasses.asdict(repl.config),
        "chat": chat.recipe(),
    }
    recorder.start(RunHeader(task=task, system_prompt=system, recipe=recipe, experiment=experiment))
    messages: list[Message] = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Task: {task}"},
    ]
    try:
        result = _run_steps(messages, repl, chat, config, recorder)
    except BaseException as e:
        outcome = Outcome.INTERRUPTED if isinstance(e, KeyboardInterrupt) else Outcome.CRASHED
        recorder.finish(RunResult(outcome, None, _steps_taken(messages)))
        raise
    recorder.finish(result)
    logger.info("run finished: %s after %d steps", result.outcome, result.steps)
    return result


def _steps_taken(messages: list[Message]) -> int:
    """Completed steps: each adds one assistant message and one observation after the two openers."""
    return (len(messages) - 2) // 2


def _run_steps(
    messages: list[Message], repl: Repl, chat: ChatModel, config: AgentConfig, recorder: Recorder
) -> RunResult:
    for index in range(1, config.max_steps + 1):
        reply = chat.complete(messages)
        code = extract_code(reply)
        if code is None:
            observation = Observation(CellStatus.ERROR, NO_CODE_BLOCK, images=(), answer=None)
        else:
            observation = repl.run(code)
        messages.append({"role": "assistant", "content": reply})
        messages.append(observation_message(observation, repl.variable_summary(), index))
        elide_old_images(messages, config.keep_images)
        recorder.step(Step(index, reply, code, observation))
        logger.info("step %d %s\n%s\n>>> %s", index, observation.status, reply, observation.text)
        if observation.done:
            return RunResult(Outcome.DONE, observation.answer, index)
    return RunResult(Outcome.STEP_LIMIT, None, config.max_steps)
