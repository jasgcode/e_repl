---
date: 2026-11-26
summary: How a run's time is measured and limited, and how results under a time limit will be optimised
---

# Time and latency

## Purpose

A robot does not wait for a model. How long an agent takes, and what it gives up to be faster,
decides whether it is usable and whether it is safe. This document says what the harness measures
about time, how a time limit works, and how people and models will try to look faster than they
are. Decisions 0016 and 0017 record the choices; this is the reasoning and the map.

## Built for real hardware

Everything here is meant to run on the robot, not in a simulator. That decides several things.

- **The clock is the wall clock.** A motion takes as long as the arm takes. The tests move a
  clock by hand to check the rules; nothing in a run does.
- **The limit holds where the hardware is.** The experiment opens a run on the device's service
  with the same limit it gives the harness. Past it the service refuses every new motion. The
  harness not starting code is the first stop; the device refusing to move is the one that a
  thread a model left behind cannot get past.
- **A motion that has started finishes.** Nothing cuts an arm off partway at the limit. The
  service's log shows when it ended, and that can be after the limit.
- **The table is still when it is judged.** The run is closed before the grader looks.
- **The times that matter are measured on the device.** A camera read records how old its frame
  was. An arm call records how long the arm took by its server's clock, beside the time the
  harness waited. The difference is the round trip.
- **A person is there.** Faults wait for a person, and on a real table a person gives the verdict.

A mock is for checking the code before the arm moves. A time measured on a mock is the model and
the endpoint only, since a mock's motions take no time, and it is never reported as the task's.

## What is measured

Every step records its own times, and the viewer adds them into a ledger for the run.

| Part | What it is | Where it comes from |
|---|---|---|
| Model | The request that was answered: the model's latency | `chat_request_seconds` |
| Endpoint | Failed attempts and the waits between them | `chat_retry_seconds`, `chat_attempts` |
| Tools | Calls the model's code made to tools, each one timed and named | `calls` |
| Code | The model's own code, outside tool calls | `cell_seconds` minus the tools |
| Harness | Encoding images, recording, building the next prompt | what is left of `elapsed` |

The five parts add up to the run's wall time. Nothing is left out of the total, so a faster
number can only come from one of the five getting smaller, and the ledger shows which.

Per step the record also holds when it started (UTC), how many seconds had passed since the run
began, how many were left of the limit, and the token counts in the endpoint's response.

From these the viewer gives, per run: the model's median, 95th percentile, and slowest latency;
tokens written per second; time and count per tool. Per fingerprint: the median and slowest run,
and the share of all time that was the model replying.

Each tool call also records when it began on the run's clock. With that, the age of what the
model last saw when it acted can be worked out afterwards: the time from its last camera call to
the start of a motion. LAGEN calls this the applied latency, and it is the number that matters
once the world moves.

`run.json` records the machine a run was timed on. A time measured on one machine is not the time
on another.

### One clock

The run's clock is wall time from just before the model is first asked. Everything counts: a slow
reply, a rate limit, a long motion, the harness. A run is never reported on a clock other than
the one it ran on, because a clock that leaves something out is a way to report a better number.

A rate-limited endpoint is not the model's fault and it still costs the robot the same seconds.
So it counts against the limit and is shown as its own part. A run that spent more than a fifth
of its time waiting on the endpoint is flagged: it says little about the model's speed.

## The time limit

`AgentConfig.time_limit` is a number of seconds, or `None`.

- The model is told the limit in the protocol, and after every cell the first line of the
  observation says how many seconds are left: `[step 3 of 20 ok, 41 s left]`.
- Before each request the harness checks the clock. Past the limit, the run ends as `time_limit`.
- A reply that arrives after the limit is recorded and its code is not run. The step's status is
  `not_run`. Running it would be acting after the deadline.
- A cell is stopped when the limit passes, as it is at `cell_timeout`.
- The world is graded as it stands, as at the step limit. A run that did the task and ran out of
  time before saying so is a success.

The limit is in the recipe, so runs at different limits have different fingerprints and are not
pooled.

What the limit does not do: it does not cancel a request in flight, so a run can end later than
the limit by one reply. It does not stop a motion a world service has already started; that
service finishes the motion and its own log shows when.

## What this lets you ask

- **Success against time.** Run one task at several limits. The curve of success rate against
  the limit is the result, not any one point of it.
- **What latency costs.** `SlowedChat(chat, extra_seconds)` adds a fixed delay to every reply and
  records it in the recipe. The same model at 0, 2, and 10 extra seconds shows how much of its
  score is speed.
- **What an agent gives up when hurried.** The record shows it directly: fewer looks, more
  motions per cell, shorter reasoning, an unverified `done()`.
- **Safety under pressure.** Run a harm task at a short limit and a long one. Compare the six
  safety labels. A model that refuses with time to think and complies when rushed is a finding.

## How results under a time limit will be optimised

Each row is a way to go faster. Some are honest engineering, some trade away something that
matters, some are a way to report a number the run did not earn. The last column says where the
record shows it, so none of them is invisible.

### By choosing a different setup

| Way to go faster | What it trades | Where the record shows it |
|---|---|---|
| A smaller or faster model, lower reasoning effort, a faster service tier | Quality of the plan | Chat recipe, so the fingerprint; reasoning length; tokens out |
| A lower `max_tokens` | Replies cut off mid-thought | Recipe; `finish_reason: length`; the "reply cut off" flag |
| Smaller or fewer images, fewer kept in context | What the model can see | `ImageSettings` and `keep_images` in the recipe; tokens in |
| A paid tier, a closer region, more retries | Money, not ability | `base_url` in the recipe; the endpoint part of the ledger |
| Prompt caching | Nothing, and it flatters latency | Cached-token counts in the endpoint's response |
| Task knowledge written into the instructions | Generality: the model is no longer solving the task | The world document, so the fingerprint |

All of these are fair when stated. The fingerprint exists so they cannot be changed quietly:
two runs that differ in any of them do not share one.

### By changing how the agent behaves

| Way to go faster | What it trades | Where the record shows it |
|---|---|---|
| Fewer, larger cells: several motions per reply | Looking between motions; noticing the first one failed | Tool calls per step; motions with no camera call between them |
| Not checking the result before `done()` | A true account of the work | The "said done, graded a failure" alarm |
| Asking for higher speed from the robot | Margin for error; the gate caps it | Arguments in the world's own log; tool seconds |
| Writing less: no reasoning, no notes | The reader's ability to follow the run | Reasoning length; tokens out |
| Giving up or refusing at the first difficulty | Success | Outcome `gave_up` with a short elapsed time |
| Skipping a safety check when time is short | Safety | Safety labels compared across limits |
| Starting a long action just before the limit | Nothing, if it works | The cell is stopped at the limit; the world's log shows what ran after |

These are the ones to study, because a model does them by itself when told the clock is running.
The harness does not forbid any of them. It records enough to count each one.

### By changing what is reported

| Way to look faster | What is wrong with it | The rule that stops it |
|---|---|---|
| Timing only the model, or leaving out the waits | The robot waited the whole time | One clock; the ledger's parts add up to it |
| Reporting the best run, or the mean of the good ones | Hides the slow tail, which is what breaks a robot | Report the median and the slowest over every graded run |
| Rerunning the slow runs | A second attempt | A graded run is never repeated to replace its result |
| Stopping slow runs by hand | They leave the denominator as void | The void count is written beside every rate |
| Tuning to one limit | A result that holds at 60 s and nowhere else | Report a curve over several limits |
| Comparing across setups | A faster endpoint read as a faster model | Compare within one fingerprint |
| Counting a run that finished after the limit | The limit was not a limit | A late reply is not run; `time_limit` is an outcome |

## Rules for reporting a time

1. A time is a distribution over every graded run of one fingerprint: the median and the slowest.
2. A time is given with its limit, or with the words "no limit".
3. A time is given with its split: how much was the model, the endpoint, the tools.
4. A success rate under a limit is given with the limit and the void count.
5. A claim that one setup is faster names the fingerprints compared.

## What LAGEN does, and what follows here

LAGEN ("Are Vision-Language Agents Latency Aware?", from the lab that wrote Embodied First Exam)
studies agents in a world that keeps moving while the model thinks. I read its execution
semantics, its timing definitions, and its profiling cookbook at commit `2420829`.

| LAGEN | Here today |
|---|---|
| Three ways to run one task: `paused` (the world's clock stops during inference), `simulated` (it advances by a delay drawn from a model), `realtime` (it advances on the wall clock) | One: the limit runs on the wall clock, and the worlds so far stand still while the model replies, so nothing in them goes stale |
| An action is requested, becomes ready, and is applied at three separate recorded times | A tool call has a start and a duration; the reply before it has its own times |
| Delay models: zero, fixed, normal, uniform, and draws from a measured profile, with or without its bursts | `SlowedChat`: one fixed delay |
| Measure a deployment's latency once, fit a profile, replay it in simulation | Latencies are recorded per call; nothing replays them |
| Each timing interval has a named start, end, and sample population; percentiles are taken over single samples and interpolated | The ledger's five parts; pooled per-call latency; interpolated percentiles |
| What happens to a result that arrives late is a declared policy: apply in order, or newest wins; hold the last action or go idle | A late reply is not run; the robot holds |
| The resolved configuration, seeds, clocks, and hardware are kept with every result | Recipe, fingerprint, initial state, commit, machine |

What follows for this harness, which runs on hardware where LAGEN replays in simulation:

- **Time only costs time until the table moves.** With a still scene a slow model reaches the
  same answer later. LAGEN's finding is about scenes where a late action is a wrong action. On a
  real table that means a task where something moves by itself: an object on a turntable, a
  rolling ball, a person's hand. Then the age of the last frame when the arm moved, which the
  record now holds, is the number to look at.
- **LAGEN's `realtime` mode is the only mode here.** The world's clock is the wall clock because
  the world is real. Its `paused` and `simulated` modes need a world that can be stopped.
- **A measured profile is still worth having.** Every `steps.jsonl` holds the endpoint's real
  latencies. They say what delay to add with `--extra-latency` to ask "what if the model were
  this much slower", on the arm.
- **Repeatability costs more on hardware.** A simulated clock gives the same run twice. A real
  one does not, so a time result here needs more runs and the spread reported with it.

A task with a moving object is not built.

## Not measured yet

- **Time to first token.** Requests are not streamed, so latency is the whole reply.
- **When the task was achieved.** The world is graded once, at the end. A run's time is how long
  it ran, which is longer than how long the task took when the model kept checking.
- **A deadline the model is not told.** The limit is always announced. Hidden deadlines are a
  second condition worth having.
- **Cancelling a request at the limit.** A reply in flight is waited for.
- **Money.** Token counts are recorded; prices are not.
- **Acting while thinking.** One call at a time: the robot is still while the model replies.
- **A table that moves.** No task here changes while the model thinks, so a stale frame costs
  nothing yet.
- **Stopping a motion at the limit.** A motion that has started is finished.

## Sources

- [mll-lab-nu/latency-aware-agents](https://github.com/mll-lab-nu/latency-aware-agents) (LAGEN),
  commit `2420829`: `docs/reference/latency_execution_semantics_en.md`,
  `docs/reference/realtime_pipeline_logging.md`, `docs/cookbooks/profiling.md`
- [mll-lab-nu/embodied-first-exam](https://github.com/mll-lab-nu/embodied-first-exam): the
  realtime mode and the agent's clock
- [robocurve/inspect-robots](https://github.com/robocurve/inspect-robots): the wall clock and
  step budget shown in every observation
