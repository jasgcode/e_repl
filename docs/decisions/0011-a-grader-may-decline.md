---
date: 2026-11-21
status: adopted
summary: A grader may return no verdict; the run it declines is void, like a crashed or interrupted one
---

# 0011. A grader may decline

## Decision

`grade` returns a `Verdict` or `None`. When it returns `None` the run keeps its outcome (`DONE` or
`STEP_LIMIT`), `result.json` holds `"verdict": null`, and the run is void under decision 0006.

## Why

- A person who grades a real table sometimes cannot say: someone bumped the scene, a camera
  dropped out, the run was fine and the setup was wrong. Asking for yes or no then records a guess.
- Decision 0006 counts the runs that have a verdict. A declined run needs no new rule and no new
  field: it is one more run without one.
- The outcome still says how the run ended, so a declined run that reached `done()` is not
  mistaken for a crash.

## Alternatives

- Raising from the grader: the run would be recorded as `CRASHED`, which says the harness or the
  world failed when a person only declined.
- A third state in `Verdict`: every reader of `result.json` would have to handle it, for the same
  effect on a rate as a missing verdict.
- No skip, as before: the operator of a real arm is pushed to pick an answer.

## Cost if wrong

A grader that forgets to return a value voids its runs without an error. The declined run carries
no reason, so the person who skipped has to remember why. A person who skips the runs that went
badly inflates the rate, as with Ctrl-C under decision 0006; the void count beside the rate is
again the only guard.
