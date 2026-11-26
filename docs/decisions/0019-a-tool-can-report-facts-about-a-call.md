---
date: 2026-11-26
status: adopted
summary: A tool object can leave facts about its last call in call_facts, and the step record keeps them
---

# 0019. A tool can report facts about a call

## Decision

A tool object may set its `call_facts` attribute to a dict while a call runs. The REPL records the
dict with that call in the step and sets the attribute back to `None`.

## Why

- The REPL times a call from outside. On hardware the times that matter are inside it: how old
  the camera frame was when it was read, how long the arm took by its server's clock.
- With a frame's age and a motion's start both in the record, the age of what the model had seen
  when it acted is a subtraction. That number is what latency costs once the world moves.
- The server-side time beside the client-side time gives the cost of the round trip.
- An attribute keeps the tool's return value what the model expects. A camera still returns an
  image.

## Alternatives

- Returning a result object with the facts on it: every tool's return type changes, and the model
  has to unwrap it.
- Reading the facts from each device's own log afterwards: possible for an arm server, not for a
  camera read in process.

## Cost if wrong

A tool that forgets to set the attribute reports nothing, silently. The name is a convention
nothing checks. Facts are whatever the tool says, with no units in the record but the key's name.
