---
date: 2026-11-21
status: adopted
summary: The system prompt is a harness-free world document followed by the harness's protocol, each recorded
---

# 0007. The system prompt is a world document followed by the harness's protocol

## Decision

The system prompt is two strings joined by a blank line:

- `world_document(instructions, tools)`: the project's instructions, then `Tools:` and the list
  from `describe_tools`. It names nothing of this harness.
- `PROTOCOL`: the cell format, what an observation holds, `show`, `done`, and the step limit.

`RunHeader` holds both, and `run.json` writes them beside the joined `system_prompt`. A project's
`AgentConfig.instructions` describes the world only. The protocol states `max_steps`, and each
observation begins `[step N of M status]`.

## Why

- Embodied First Exam runs several harnesses on one task because its task text is the same bytes
  for each of them and says nothing of any harness. See
  [the design](../designs/2026-11-21-embodied-first-exam-principles-design.md), principles 2 and 6.
- The world document is the file another harness would be handed. Building it as its own string
  makes that a function call, with no text to cut out of a blended prompt.
- With the parts recorded, a reader of `run.json` can tell a change to the world's description
  from a change to the harness's protocol.
- The model was never told its step limit. A budget the agent cannot see cannot be planned for.
- `system_prompt` stays in `run.json` because it is the bytes that were sent.

## Alternatives

- The blended prompt as it was, with the tool list inside `PROTOCOL`: the tools are facts about
  the world, and they could not be handed to another harness without the cell format around them.
- The protocol first and the world second: the instructions opened the prompt before this
  change, and keeping that order leaves the moved tool list and two new sentences as the only
  differences.
- Recording only the two parts: a reader would have to know the join rule to rebuild what the
  model was sent.
- The step budget in the observation header alone: the model would learn its limit only after
  spending a step.

## Cost if wrong

Every fingerprint moves once, so runs from before this change do not pool with runs after it.
`run.json` holds the prompt text twice.
