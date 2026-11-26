---
date: 2026-11-26
status: adopted
summary: A run can have a wall-clock limit the model is told about; everything counts and no code runs after it
---

# 0017. A run can have a time limit the model is told

## Decision

`AgentConfig.time_limit` is seconds of wall clock from just before the model is first asked, or
`None`. Everything counts against it: replies, waits on the endpoint, cells, the harness.

The protocol states the limit, and each observation's first line gives the seconds left. Past the
limit no request is sent and no code is started. A reply that arrives late is recorded with status
`not_run`. A cell running when the limit passes is stopped. The run ends as `time_limit` and the
world is graded as it stands.

`SlowedChat(chat, extra_seconds)` adds a fixed delay to every reply and puts it in the recipe.

## Why

- A step limit counts the harness's own unit. A robot's constraint is seconds.
- One clock that counts everything cannot be chosen to flatter a result. A rate limit costs the
  robot the same seconds as a slow model; it is shown apart in the ledger and still counted.
- A model that cannot see its clock cannot be studied under time pressure. Telling it makes
  hurrying, and what is dropped when hurrying, something a run shows.
- Code that starts after the deadline is the easiest way to beat a limit, so none does.
- The limit is in the recipe: runs at different limits are different setups and are not pooled.
- A fixed added delay turns "how much of this score is speed" into an experiment that needs no
  slower model.
- See [the design](../designs/2026-11-26-time-and-latency-design.md).

## Alternatives

- A clock that stops while the endpoint is rate limited: fairer to the model and no longer the
  time the robot waited. The ledger keeps both facts.
- A limit the model is not told: a different condition, worth adding beside this one.
- Ending the run for the model at the limit and grading later: what ran in between would count.
- Cancelling the request in flight: exact, and it needs a deadline in the chat protocol.

## Cost if wrong

A run can end later than its limit by one reply, since a request is waited for. A world service
finishes a motion it has started. Naming the limit changes the prompt, so a limited run and an
unlimited one differ in more than the limit. Wall time depends on the machine and the network, so
a time measured here is not the time elsewhere.
