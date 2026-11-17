---
date: 2026-11-17
status: adopted
summary: The model's only action is running a cell in a persistent Python REPL
---

# 0001. The model's only action is running a cell in a persistent Python REPL

## Decision

The model acts by writing one fenced code block per turn. The harness runs it in a namespace that
persists across turns. Tools, cameras, and sub-model calls are Python objects in that namespace.
The observation is the printed output (truncated), the value of a trailing expression, the images
passed to `show()`, and a one-line summary of each variable.

## Why

- The model composes capabilities with loops, conditionals, and numpy in one turn, where a
  structured tool call does one thing per turn.
- Bulky data (frames, depth maps, point lists) stays in variables. The context holds summaries.
- A new capability is a new object with a docstring. Nothing in the harness changes.
- The fenced-block format works with any chat model, including free endpoints with no
  tool-calling support.

## Alternatives

- Structured tool calls (JSON schemas): one call per turn, and every result enters the context.
- A Jupyter kernel in a subprocess: isolates crashes, but tools holding serial ports and camera
  handles would have to live in the kernel or be proxied to it.
- Recursive sub-agents with their own REPLs: `llm_query` inside one REPL covers the cases met so
  far.

## Cost if wrong

Model code runs in the harness's process with its rights: a cell can block, exhaust memory, or
reach around a tool. Safety limits therefore sit inside the tools, and a process boundary would
mean moving `Repl.run` behind a kernel, which the `Observation` type already allows.
