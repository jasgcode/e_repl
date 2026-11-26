---
date: 2026-11-26
status: adopted
summary: give_up(reason) ends a run beside done(), so a false claim of success can be told from an honest stop
---

# 0014. give_up() ends a run without a claim of success

## Decision

The REPL has a third builtin, `give_up(reason)`. It ends the run with outcome `gave_up`, and
the reason is recorded as the run's answer. `done()` now means one thing: the model says the task
is complete. The protocol tells the model to call `give_up` when the task cannot or should not be
done, and never to call `done()` for a task it did not complete. A reason is required.

Both endings are graded as before. The viewer raises an alarm when the ending and the verdict
disagree: `done` with a failure, or `gave_up` with a success.

## Why

- With one way to stop, the protocol told the model to call `done()` when the task was "complete
  or clearly impossible". A run that ended `done` and was graded a failure could be a false claim
  or an honest "I could not", and the record could not say which.
- A model's account of its own work is something to measure. Separating the two endings makes
  "said it was done and it was not" a count over run directories.
- `give_up` is also where a model declines a task it should not do. A refusal is then an outcome
  in `result.json`, with a reason, and not text to search for.
- inspect-robots gives its agent the same pair, `done` and `give_up`, and scores neither as a
  verdict.

## Alternatives

- One `done(answer, success=...)`: the claim becomes an argument a model can leave at its
  default, and the default would have to be one of the two.
- Parsing the answer text for words like "could not": a guess about the thing being measured.
- A separate `refuse()` beside `give_up()`: cannot-do and should-not-do are different findings,
  and the reason already carries which. One more ending can be added when a study needs the count.

## Cost if wrong

The protocol text changes, so fingerprints move once more. A model can still call `done()` for a
task it did not complete; that is the behaviour the split makes visible, and nothing prevents it.
A model that gives up at the first difficulty lowers its success rate where it used to run out of
steps.
