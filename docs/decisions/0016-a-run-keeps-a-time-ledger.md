---
date: 2026-11-26
status: adopted
summary: Every step times the model apart from the endpoint and times each tool call, so a run's time splits five ways
---

# 0016. A run keeps a time ledger

## Decision

A `Reply` holds `request_seconds`, the latency of the one request that was answered, beside
`seconds`, which includes failed attempts and the waits between them. The REPL times every call
the model's code makes to a tool and records its name, when it began, its seconds, and what it
raised. A step records these with the seconds since the run began, and `run.json` records the
machine the run was timed on.

The viewer adds them into a ledger of five parts that sum to the run's wall time: the model, the
endpoint, the tools, the model's own code, and the harness.

Tools are timed by replacing each public method on the tool object with a timed one. The object
stays the same object, so the model sees no wrapper. A tool calling its own methods counts once.
A tool whose methods cannot be replaced is refused when the REPL is built.

## Why

- A total says a run was slow and not why. On a rate-limited endpoint most of a run can be
  waiting; without the split that reads as a slow model.
- "The time it took to do something" is a tool call. Before this, a motion was text in a cell,
  with no duration of its own.
- Parts that add up to the total leave nowhere to hide time: a faster number has to come from one
  of the five, and the ledger shows which.
- Timing in the REPL covers every tool with no code in the tool. A tool that timed itself would
  be one more thing each project could get wrong or leave out.
- See [the design](../designs/2026-11-26-time-and-latency-design.md).

## Alternatives

- A proxy object around each tool: the model would see another type name and `isinstance` would
  change.
- Timing only in a world service's log: exact for that world, and nothing for an in-process tool
  or a sub-model call.
- Python's profiling hooks: every call in the cell, most of them noise.

## Cost if wrong

The REPL changes the tool objects it is given, and the experiment's own calls on them pass
through the timer too. A tool built on `__slots__` or a frozen dataclass cannot be used until it
is changed. Latency is the whole reply, since nothing is streamed.
