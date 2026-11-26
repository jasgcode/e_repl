---
date: 2026-11-26
status: adopted
summary: Improvement loops run as campaigns; each round records the loop's judge beside the world's held-out grade
---

# 0022. Campaigns record a loop's judge beside a held-out grade

## Decision

`run_campaign(loop, judge, config, library, practice, recorder)` runs rounds of an improvement
loop from parts that swap independently. A `Loop` (protocol: `kind`, `recipe()`, `task(index)`,
`held_out(library)`) says what each round asks for and grades what was kept. A `Judge` (protocol:
`kind`, `recipe()`, `judge(run, library)`) decides what is kept. The practice function holds the
world and the model. The record takes each part's kind and recipe from the object that ran.

Each round is one practice run, recorded as any run is, in which the agent holds a `Library` of
named texts that outlives the round. Then:

- the **judge** decides whether the library is kept; its kind is one of `sealed`, `self`,
  `model`, `agent_written`, `person`;
- the loop's **held-out** grade scores the same library on conditions the judge did not use;
- a rejected or undecided round's library is restored to what the last kept round left;
- the next round's task carries what `CampaignConfig.briefing` allows: nothing, the judge's
  reason, or the judge's reason and the held-out grade.

`judge_agreement` counts rounds in four cells. *Judge passed, world failed* is the loop's judge
being fooled or gamed. `LoopKind` names the loop: `program`, `practice`, `skills`, `scaffold`.
Loops that train weights, need a world model or a simulator, or change a body are not supported;
the [design](../designs/2026-11-26-improvement-loops-design.md) maps every row.

## Why

- The failure reported for nearly every self-improvement loop is the judge passing what the world
  fails. Recording both on every round makes that a rate instead of an anecdote.
- The held-out grade follows decision 0005: the world grades, the model's side decides nothing.
  The judge is the loop's own mechanism, so it is recorded as what is under study, never as truth.
- A library is a tool, so the agent's work stays code in the namespace (decision 0001), and the
  next round's agent finds it by reading it. Only the campaign can revert it, because `snapshot`
  and `restore` are functions outside the tool.
- The briefing is a config value because what the next round is told changes what it does
  (decision 0003's rule). Showing the held-out grade is offered so its effect can be measured.
- Practice runs stay ordinary runs. Rates, the viewer, and the time ledger apply to them unchanged.
- Loops and judges are separate objects so one can change while the other stays: the question
  "does this loop's judge get fooled" is asked by holding the loop and swapping the judge. Their
  kinds come from the objects, so a label in a config cannot disagree with what ran.

## Alternatives

- Carry the work in the prompt: the next round's system text holds the last round's program. It
  changes the fingerprint every round and puts the work in the context whether the agent looks or
  not.
- One function per loop that does everything: shorter, and a judge could not be swapped
  without copying the loop.
- One grader for both: the loop keeps what the world passes. Nothing would then be measured about
  the loop's own judge, which is the part that fails.
- Support every row of the table, with a training loop and a simulator. That is a training
  framework, and decision 0021 leaves benchmarks and training elsewhere.

## Cost if wrong

Held out runs every round, which on hardware is the expensive part. A judge that needs the
transcript rather than the library (a `model` judge) has only the `Practice` it is handed. The
`practice`, `skills`, and `scaffold` loops have names and no toy world yet.
