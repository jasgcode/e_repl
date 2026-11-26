---
date: 2026-11-26
status: adopted
summary: Several agents can work on one task by taking turns in the one loop, sharing its world, limits, and record
---

# 0020. Several agents can share a run

## Decision

`run_team(task, members, ...)` runs two or more agents on one task. A `Member` is a name, a chat
model, the tools it holds, and an optional role. `swarm(count, chat, tools)` makes equal members
on one model. `run_agent` is unchanged, and a run with one agent has the prompt and fingerprint
it had.

- Each member has its own conversation and its own REPL namespace. Members that hold the same
  tool objects share a world.
- They take turns in the order given: one reply and one cell each.
- `tell(message)` sends text to every other member. It is shown at the top of the receiver's
  next observation, and both ends are recorded in the step.
- The step limit and the time limit are the team's. So are the one grader, the one harm judge,
  and the one recorder; each step names the agent that took it.
- Any member's `done()`, `give_up()`, or `refuse()` ends the run for all.

## Why

- A second agent is the cheapest check on the first that is not the first: one acts and one
  watches, or several look at a problem from different prompts. Whether that helps, and what it
  costs in time, is now something a run can measure.
- One loop for one agent or many keeps the rules in one place. The time limit, the late reply
  that is not run, the grading, and the record are the same code for a team.
- Turns, not threads. A device serves one call at a time, a cell captures the process's output,
  and a run that is the same twice can be studied. Agents acting at once would give up all three.
- Any member can end the run so that any member can refuse. A team where one agent's refusal can
  be outvoted is less safe than that agent alone.
- Messages are delivered, not fetched, because a model that has to remember to check for mail
  mostly does not. They are recorded on both sides, so what was said and what was heard can be
  compared.
- Shared limits make a team comparable with one agent: the same steps and the same seconds,
  spent by more heads.

## Alternatives

- Agents running at the same time in threads or processes: closer to a real swarm, and the order
  of their actions would differ from run to run. Each process with its own client of one device
  service is the way to add it.
- A lead agent that hands work to the others: a hierarchy is one arrangement of roles, and roles
  are text a member is given. The loop does not need to know.
- A shared namespace: no message passing needed, and no way to say which agent knew what.
- Ending by agreement: safer against a hasty `done()`, and a single refusal would no longer stop
  the run.

## Cost if wrong

A member waits a full round before it acts again, so with many members each one sees an old
world. A team spends the step budget faster per decision. Members on one model with one prompt
tend to agree with each other, which looks like checking and is not. Messages are text only:
an agent cannot hand another an image. The fingerprint of a team run covers every member's
prompt, so one changed role is a different setup.
