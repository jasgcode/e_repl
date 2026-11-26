---
date: 2026-11-21
status: adopted
summary: The world grades each run and the verdict is recorded with the result; done() decides nothing
---

# 0005. The world grades each run and the verdict is recorded with the result

## Decision

`run_agent` requires a `grade` callable that takes no arguments and returns a
`Verdict(success, reason)`. The loop calls it once, after the last cell, when a run ends as `DONE`
or `STEP_LIMIT`. The verdict is a field of `RunResult` and is written to `result.json`. A run that
is interrupted or crashes has no verdict, and a grader that raises makes the run `CRASHED`.

A grader is a function the experiment holds. It is never a tool and never a public method of one.

## Why

- `Outcome.DONE` means the model called `done()`, and the answer it passed is a claim. Before this
  change no file in a run directory said whether the claim held.
- `docs/README.md` defines a success rate as a count over run directories with one fingerprint.
  `toy_pick` checked the shelf for its exit code and wrote the check nowhere, so the directories
  could not answer the question.
- A run that reaches the step limit may have achieved the task without saying so. Grading the
  world as it stands counts that run for what it did.
- `reason` lets a failed run's `result.json` say why without a reader opening `steps.jsonl`.
- The argument is required. With a default, a run made without a grader would look graded or
  leave no verdict, and the loop would carry an optional path.
- The principle is taken from Embodied First Exam, whose service writes the verdict and whose
  verifier only transcribes it. See
  [the design](../designs/2026-11-21-embodied-first-exam-principles-design.md), principle 4.

## Alternatives

- A `bool`: enough for a rate, but a failure then says nothing about itself.
- Grading in the experiment after `run_agent` returns, into a second file: a path around the
  recorder, and `result.json` would be written before the verdict is known.
- A `verdict()` method on the tool: `describe_tools` lists every public method of an object in the
  namespace, so the model would be shown the success check and could call it.
- Grading interrupted runs too: Ctrl-C on a robot is a person stopping a run, and reading
  the world from the handler of that interrupt delays the stop. Whether such a run counts as a
  failure is a separate question about counting.

## Cost if wrong

`result.json` gains a key, and run directories written before this change lack it. Every caller
of `run_agent` must supply a grader, including a project whose task has no end state a program
can check; that project's grader has to say so in its reason.
