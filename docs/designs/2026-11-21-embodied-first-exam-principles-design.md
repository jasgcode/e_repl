---
date: 2026-11-21
summary: What e_repl takes from Embodied First Exam, where the harness seam is at the world and the world grades
---

# Design principles from Embodied First Exam

## Purpose

Embodied First Exam (EFE) runs unmodified Codex, Claude Code, and Antigravity CLI against the same
1,156 robot tasks. This document records how its code makes that work, the principles e_repl should
take from it, and the change each one implies here.

Nothing below is implemented. A change that chooses between options gets its own decision in
`docs/decisions/` when it is made; the last two sections list the changes and the choices that are
still open.

## What was read

The repository [mll-lab-nu/embodied-first-exam](https://github.com/mll-lab-nu/embodied-first-exam)
at commit `e6db573` (2026-11-20):

- `docs/protocol/README.md`: the task protocol and its eight binding invariants (H.1 to H.8)
- `docs/protocol/running.md`: official runs, infrastructure failures, results, the agent's time
- `docs/protocol/standard_mode.md`: Part 1 (the base wire protocol) and Part 3 (realtime)
- `embodied_first_exam/standard/rcb_client.py` and `toy_server.py`: the shared client and the
  smallest complete service
- one reference task, `tasks/robotwin-2/robotwin_2_0_stack_blocks_two-standard/`, in full

Not read: `rcb_limited.py` (the shared service layer), Harbor's own source, and the other tasks.
The paper's figures in principle 11 come from the arXiv HTML of
[2611.12177](https://arxiv.org/abs/2611.12177) and were not checked against the PDF.

## How EFE runs several harnesses

It has no harness code. `running.md` makes that a condition of an official result:

> Harbor's native agents (`-a codex`, `-a claude-code`, ...), configured through environment
> variables; no agent adapter, no gateway that holds credentials or changes requests.

and, in the same list, "nothing added to the instruction (no prompt template, no extra
instruction)". Harbor installs the stock agent CLI into the task's container and hands it one
file. Everything that makes several harnesses work is on the task's side of the line.

A standard-mode agent gets three things:

1. `instruction.md`: the task, the rules, a code sample of the client, every unit, frame, and
   action layout, and the time limit.
2. `/opt/tools/rcb_client.py`: one client module, in a `python:3.12-slim` image that holds no
   simulator.
3. `r.spec()`: action dimensions, joint limits, cameras, the world frame, and the budgets, as data.

```
agent container (no simulator)                   sim sidecar (the agent cannot open it)
  the stock harness CLI                            the simulator
  instruction.md                                   the clock and the budgets
  rcb_client.py  <------- HTTP sim:8000 ------->   /spec /status /observe /submit
  whatever files the agent writes                  the episode's record and its verdict
```

A harness therefore needs two abilities: read a file and run Python. That is the whole
compatibility requirement, and it is why Codex and Claude Code needed no accommodation.

## Where e_repl stands

In EFE's terms e_repl is one harness. `run_agent`, `PROTOCOL`, `show()`, and `done()` play the part
Codex's CLI and its built-in prompt play there. The comparison that matters is between EFE's task
side and what e_repl calls the world: the dict of tools passed to `Repl(...)`.

That world lives inside the harness's process. Only `run_agent` can reach it, so no other harness
can drive the same tools, and the model's code can write the world's state. In the toy,
`shelf.picked = "red"` in a cell satisfies the check in `experiments/toy_pick.py` without a pick.

## Principles

Each entry gives what EFE does, where e_repl is today, and the change.

### 1. The world carries no harness code

**EFE.** The robot is a service with a handful of endpoints, and every other path is 404. The agent's
side is a client module and a text file. Invariant H.3: "The agent's controller runs in the
agent's own container; no agent code runs next to the simulator."

**e_repl.** The world is Python objects in the harness's process. `AGENTS.md` already says the REPL
is not a sandbox and that a safety limit belongs inside the tool; it does not say the tool can be a
client of something else.

**Change.** Allow, and show once, a world that runs as its own process with a small client object
in the namespace. The client keeps the method names, argument order, and units of the in-process
tool, which is the mock and real rule applied a third time. This gives three things: another
harness can drive the same world by importing the client, a cell cannot write the world's state or
its verdict, and the world keeps running (an arm keeps holding its pose) when the harness crashes.
The cost is a process to start, a wire format, and latency on every call. Decision 0001 is
untouched: the model's one action is still a cell.

### 2. The task text is the same bytes for every harness

**EFE.** `instruction.md` is harness-free, and an official run adds nothing to it. What a harness
says to its model about its own mechanics is the harness's business.

**e_repl.** `system_prompt()` blends three things into one string: the project's `instructions`,
the tool list from `describe_tools`, and the harness's own protocol (the ` ```repl ` block format,
`show`, `done`, truncation). `RunHeader` stores the blend, and the fingerprint hashes it.

**Change.** Build two strings and concatenate them: a world document (instructions plus the tool
list) and the protocol text. Record both in `RunHeader`. The world document is what another harness
would be handed as its instruction file. Moving the tool list out of `PROTOCOL` changes the prompt's
bytes, so fingerprints change once.

### 3. Nothing enters the context unless the agent's code puts it there

**EFE.** From the instruction: "Nothing is put into your context automatically. Images and videos
come back as arrays/bytes in a Python object; what you look at and where you store it is your
code's decision." `res.save(dir)` writes JPEGs and an MP4, and the harness's own file viewer does
the looking. That is how images cross harnesses whose image handling differs.

**e_repl.** `show()` is the same rule: a tool returns an array and the model's code decides to look.

**Change.** None to the harness. Add a line to the tool rules in `AGENTS.md`: a tool returns data
and never calls `show()` itself. A client written under principle 1 also gives its results a
`save(directory)`, since a harness without `show()` looks at files.

### 4. The world owns the verdict

**EFE.** Invariant H.4: the service writes `final.json` from the episode it ran, and the verifier
only transcribes it. The verdict is on disk before any video. At the cutoff every process the agent
left running is killed before the service is told to finalize, and from that signal on the service
refuses every request that would advance the simulator. The agent's `done` decides nothing.

**e_repl.** `RunResult` is `(outcome, answer, steps)`, and `Outcome.DONE` means the model called
`done()`. `toy_pick.py` compares `shelf.picked` with the color for its exit code and writes the
result nowhere. `docs/README.md` defines a success rate as "a count over directories with one
fingerprint", and today the directories do not say which runs succeeded.

**Change.** The experiment passes `run_agent` a grader that the world owns. The loop calls it once
after the last cell, on every outcome where the model's actions stand (`DONE`, `STEP_LIMIT`), and
the verdict goes into `RunResult` and `result.json`. A crashed run gets no verdict (principle 8).
Grading happens after the world is still: a tool that moves hardware returns only when the motion
has ended, and nothing the model started keeps acting afterwards.

### 5. Two fixed agents admit a task

**EFE.** A task is valid only when `nop` scores 0 and the oracle scores 1 on every oracle seed. The
oracle is the reference solution run through the same `-a` slot as Codex, so it exercises the path
a real run takes. A new benchmark merges only after "one adversarial trial (an agent told to obtain
reward without solving the task) scored 0 on its tasks."

**e_repl.** `test_a_scripted_model_picks_the_red_box_and_the_run_is_on_disk` is the oracle half:
`ScriptedChat([LOOK, FIND, PICK])` through `run_agent` on three seeds. There is no do-nothing test
and no adversarial one.

**Change.** Add both for the toy, written against the grader from principle 4: a model whose only
cell is `done("picked the red box")` must be graded a failure, and so must one whose cell assigns
`shelf.picked`. Add the rule to `AGENTS.md`: a task ships with its oracle script and a do-nothing
test. In process, a read-only property stops the accidental write and not a determined one; only
the boundary in principle 1 does that.

### 6. The budget is the world's, and every response carries what is left

**EFE.** "Steps are unlimited; your wall-clock budget is the limit." Every response carries
`meta["time_left_s"]`, counted from the agent's first request so it never runs out before the
harness's own cutoff. The instruction states the limit and tells the agent to read the clock
"before you decide that time is short, never guess".

**e_repl.** `max_steps` is the budget, and it counts e_repl's cells, a unit no other harness has.
The model is never told the number: `PROTOCOL` does not mention it, and the observation header is
`[step 3 ok]`.

**Change.** Tell the model its budget in the header of each observation (`[step 3 of 8 ok]`), which
is cheap and changes the prompt. A world with a real limit (seconds of arm time, one pick) states it
in its instructions and returns what is left from the tool that spends it.

### 7. Bounds live inside the call

**EFE.** Three behaviors, each stated in the spec:

- A refused call runs nothing. The error table is headed "Errors (no physics executed)", and a `dt`
  that is not a multiple of the control step is "rejected (400), never rounded".
- An adjusted call says so. Out-of-limit joint targets are clipped and listed in
  `meta["clipped"]`.
- A call that runs too long "stops at the next control step and returns normally", with
  `meta.stopped` and `executed_rows` below the rows sent, and the response says where the robot is.

**e_repl.** `cell_timeout` interrupts from outside: `_time_limit` raises `TimeoutError` from a
signal handler at whatever line the tool is on. The tool is not asked to stop and does not report
how far it got. The first behavior is already a rule here ("a tool that refuses a call raises").

**Change.** Add the other two to the tool rules: a tool never changes an argument silently, and a
tool that can stop partway returns or raises with how much ran and the state it stopped in. Document
`cell_timeout` as a backstop for runaway model code, set longer than the longest call any tool
allows, so a tool's own bound always fires first.

### 8. An infrastructure failure needs positive evidence

**EFE.** A void attempt is moved aside, never counted, and rerun with the same configuration. "It
needs positive evidence: a trial is not voided because it scored 0." The other half: "What the
agent's commands caused stays a result, never an infrastructure failure (a rerun would be a free
second attempt)." One function, `retry_policy.py`, implements the rule.

**e_repl.** `Outcome.CRASHED` is "an exception outside the model's code", which is the right line.
No rule says how such a run counts. A fault inside a tool that is not the model's doing (a dropped
serial port) reaches the model as an ordinary cell error and is recorded as one.

**Change.** State the counting rule in `docs/README.md`: runs with no verdict are void and leave
the denominator. Decision 0006 makes interrupted runs void as well as crashed ones, which departs
from EFE's rule above. Whether a tool can declare a fault that is not the model's is an open
decision below.

### 9. A result carries the task's identity and the configuration's

**EFE.** Each task has a `DIGEST`, a sha256 over its files: "a changed file is a changed task."
Rankings "compare trials only within one comparable configuration" (protocol version, mode, cutoff,
reset budget, slowdown, network). The seed is recorded with each result and is not part of that key.

**e_repl.** The fingerprint is the configuration half and covers the instructions, the tool list,
both configs, and the chat recipe. The task string is in `run.json`. The world's initial state is
not: the toy's `--seed` appears only in the recorded command line.

**Change.** `RunHeader` gains the world's own description of its initial state (for the toy, the
seed), supplied by the experiment and left out of the fingerprint, as the task is. A run can then
be rebuilt from `run.json` alone, and a rate can name the seeds it covers.

### 10. A copied layer is checked byte for byte

**EFE.** The shared service layer and client are "copied unchanged" into every task, and
`ci_checks/check-tooling-identity.sh` fails when a copy differs from its source.

**e_repl.** A project's `harness/` is a copy of this one. simple_101's is identical to commit
`48d076d` today, and nothing checks that it stays so.

**Change.** Ship a digest of `harness/` with the template and a test that recomputes it. A project
that edits its copy then fails its own tests until it records the new digest, which turns a silent
difference into a deliberate one.

### 11. The harness is part of the measurement

**EFE.** The paper reports that switching harness moves success by up to 12 points: one model
scored 34% in Codex and 22% in Claude Code on the same 456 tests. Some of the gap is policy, such
as what a harness does when a content filter blocks a reply (one ends the session, one resends, one
asks the model to continue). So a run id names the harness with the model and the effort.

**e_repl.** The recipe holds the configs and the chat model. Policies that live in code are covered
only by the commit: what `extract_code` does with several blocks, the reply to a turn with no code
block, `strip_thinking`. Two commits with different policies share a fingerprint.

**Change.** When the policy the README leaves open is chosen, make it a value in `AgentConfig`
so the fingerprint holds it. Apply the same test to any later policy: if it changes what the model
sees or which of its code runs, it is a config value.

### 12. Say what the world does while the model thinks

**EFE.** Standard mode advances the simulator only inside a call. Realtime mode never stops the
clock: the robot holds its last targets while the agent thinks, a new plan replaces the old one
from a stated step, and the instruction says all of this.

**e_repl.** A real arm is always in the realtime case, and nothing asks a project to describe it.

**Change.** Add to the tool rules: the instructions say what the hardware does between cells
(holds its pose with torque on, or goes limp) and what a call does to a motion still under way.

## What not to take

- Containers, egress allowlists, and the usage gateway. They defend a public leaderboard against a
  capable agent; e_repl runs a model the operator chose on the operator's bench.
- Crash recovery by journal replay. A real arm cannot be replayed to a state.
- The resource monitor and the RCB1 binary frame format.
- Privileged mode, where the agent shares a container with the simulator.
- Harbor itself. e_repl's loop is the thing under study, so it stays.

## Changes, in order

| # | Change | Touches | Principle | Decision |
|---|---|---|---|---|
| 1 | The world grades; the verdict is in `result.json` | `records.py`, `loop.py`, `toy_pick.py` | 4 | yes |
| 2 | Do-nothing and adversarial tests for the toy | `tests/`, `toy.py` | 5 | no |
| 3 | Counting rule: runs with no verdict are void | `docs/README.md` | 8 | yes |
| 4 | World document and protocol as two recorded strings | `loop.py`, `records.py` | 2 | yes |
| 5 | The observation header shows the budget | `loop.py` | 6 | no |
| 6 | The world's initial state in `RunHeader` | `records.py`, `toy_pick.py` | 9 | yes |
| 7 | Tool rules: no silent adjustment, partial calls report, no `show()` in tools, say what happens between cells | `AGENTS.md` | 3, 7, 12 | no |
| 8 | A digest of `harness/` and its test | `harness/`, `tests/` | 10 | yes |
| 9 | The toy shelf as a service with a client | `src/project/`, `experiments/` | 1 | yes |

Changes 1 to 3 make a success rate something the run directories can answer, and they are small.
They are done: decisions [0005](../decisions/0005-the-world-grades-each-run.md) and
[0006](../decisions/0006-a-success-rate-counts-graded-runs-only.md).
Changes 4 to 6 are done too: decisions
[0007](../decisions/0007-the-prompt-is-a-world-document-and-a-protocol.md) and
[0008](../decisions/0008-a-run-records-the-worlds-initial-state.md).
Changes 7 and 8 are done as well: the tool rules are in `AGENTS.md`, and the digest is decision
[0009](../decisions/0009-the-harness-records-a-digest-of-its-files.md). Change 9 is done for the
toy: decision [0010](../decisions/0010-a-world-can-run-as-its-own-process.md).
Changes 4 to 6 each alter the prompt or the header, so they are best done together: fingerprints
move once. Change 9 is the one that lets another harness drive a project's hardware, and it is the
largest.

## Open decisions

1. **The verdict's shape.** A bool, or a record with a reason and the world's own measurements.
   Decided in 0005: `Verdict(success, reason)`.
2. **Is Ctrl-C a result?** Decision 0003 calls it "a labelled failure worth keeping", which fits
   EFE's rule that what the agent caused is a result. A person can also stop a run for a reason
   that is not the model's. Decided in 0006: void, like a crash.
3. **A tool's own faults.** Either a named exception that `Repl.run` lets through, ending the run
   as `CRASHED`, or the world records the fault and the grader reports the run void.
4. **Steps or seconds.** Whether `max_steps` stays the harness's budget once a world can state its
   own.
5. **The transport for a world service.** HTTP on localhost as EFE does, or files as simple_101's
   camera server does. Decided in 0010: HTTP for calls; files remain right for frame streams.
6. **Where a safety gate sits.** Under principle 1 it belongs on the service side, in the class
   that drives the hardware. In a project whose gate the user owns, moving it is the user's change.

## Sources

- [mll-lab-nu/embodied-first-exam](https://github.com/mll-lab-nu/embodied-first-exam), commit
  `e6db5737f72627597713a96087f27b1b17b06063`
- [Task protocol](https://github.com/mll-lab-nu/embodied-first-exam/blob/main/docs/protocol/README.md):
  §2 (validation), Appendix H (invariants)
- [Running](https://github.com/mll-lab-nu/embodied-first-exam/blob/main/docs/protocol/running.md):
  "Official runs", "Infrastructure failures", "Results"
- [Standard mode](https://github.com/mll-lab-nu/embodied-first-exam/blob/main/docs/protocol/standard_mode.md):
  "Errors (no physics executed)", "The agent's clock", "Per-call bounds"
- [`rcb_client.py`](https://github.com/mll-lab-nu/embodied-first-exam/blob/main/embodied_first_exam/standard/rcb_client.py)
  and [`toy_server.py`](https://github.com/mll-lab-nu/embodied-first-exam/blob/main/embodied_first_exam/standard/toy_server.py)
- [Embodied First Exam, arXiv 2611.12177](https://arxiv.org/abs/2611.12177)
