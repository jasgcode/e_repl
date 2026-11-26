---
date: 2026-11-26
status: adopted
summary: A step records the model's reasoning, the endpoint's whole response, the times, and all the model was shown
---

# 0012. A step records what cannot be worked out later

## Decision

`ChatModel.complete` returns a `Reply`: the answer text, the model's reasoning, the endpoint's
response body verbatim, the call's wall time, and its attempts. Each line of `steps.jsonl` holds
those with the step's start time, the cell's wall time, and the variable summary the model was
shown. `run.json` holds the run's start time and `result.json` its finish time, in UTC.

Reasoning comes from `<think>` blocks and from the endpoint's `reasoning_content` or `reasoning`
field. It is recorded and never sent back to the model. A `<think>` that never closes is a reply
cut off mid-thought, and everything after it is reasoning.

## Why

- The records are for studying what a model did, later. Reasoning was removed before anything
  was recorded, and it is the one part of a run that cannot be rebuilt from the rest.
- The response body holds the model the endpoint really served, why the reply stopped (`length`,
  `content_filter`), and the token counts. None of these are in the reply text.
- The variable summary is part of every observation the model was sent. A record that claims to
  hold what the model saw was missing it.
- Times make a run comparable with the world's own logs: a robot's service log, a camera's frames.
- A reply cut off inside its thinking used to keep the `<think>` text as the answer, so a code
  block the model was only considering could be taken as its cell and run.

## Alternatives

- A second file for responses: `steps.jsonl` would stay small, and a reader would have to join
  two files by step to see one step.
- Recording the whole request each step: exact, and each line would repeat every earlier one. The
  request is the system prompt, the earlier steps, and the image-keeping rule in the recipe.
- Sending reasoning back to the model: some endpoints want it, and it would change what every
  model sees. This decision changes no prompt.

## Cost if wrong

`steps.jsonl` roughly doubles in size, since the response repeats the reply. Calls made through
`llm_query` are still not recorded. The recorder runs in the model's process, so these records
are what the harness wrote, with no proof that a cell did not edit them afterwards.
