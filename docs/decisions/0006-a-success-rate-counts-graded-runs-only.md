---
date: 2026-11-21
status: adopted
summary: A success rate counts graded runs only; crashed and interrupted runs are void and reported beside it
---

# 0006. A success rate counts graded runs only

## Decision

A run counts toward a success rate when its `result.json` holds a verdict. The rate is the runs
whose verdict is a success over the runs that have a verdict, among the directories of one
fingerprint.

A run that ended `CRASHED` or `INTERRUPTED` has no verdict, and neither has a directory with no
`result.json`. These runs are void: they stay on disk, they are left out of both numbers, and
their count is written beside the rate (`7 of 10, 2 void`).

A void run may be repeated with the same recipe, task, and initial state. A graded run is never
repeated to replace its result.

Decision 0003 stands: every run is still recorded. It calls an interrupted run "a labelled failure
worth keeping"; the record and its label stay, and under this decision the run is not a failure in
a rate.

## Why

- A crash comes from outside the model's code (the endpoint, the recorder, the grader), so it says
  nothing about the model. `Outcome.CRASHED` is the positive evidence that Embodied First Exam
  asks for before it voids an attempt.
- A person presses Ctrl-C for reasons the record cannot tell apart: the model was going wrong,
  someone stepped into the workspace, a cable caught. Decision 0005 does not grade such a run, so
  calling it a failure would be a guess.
- The rule is "has a verdict", which decision 0005 already made true in the code. Counting needs
  one test on one field, and there is no second list of outcomes to keep in step with the loop.
- The void count beside the rate keeps stopped and crashed runs visible to a reader.
- Repeating a graded run to replace it would be a second attempt at the same task.

## Alternatives

- Interrupted runs count as failures: Embodied First Exam's rule is that what the agent's actions
  caused stays a result, and this closes the gap named under "Cost if wrong". It also charges the
  model for every stop that was not its doing.
- Asking the person for a reason at Ctrl-C: a prompt in the middle of an emergency stop. A reason
  written afterwards could be added to the record later without changing this rule.
- Counting every directory: an endpoint outage would then lower a model's rate.

## Cost if wrong

A person who stops the runs that are going badly removes failures from the denominator, and the
rate reads higher than the model earned. The void count is the only guard: a rate with many
interrupted runs beside it has not been measured. Every run is kept, so a rate can be recomputed
with interrupted runs as failures at any time.
