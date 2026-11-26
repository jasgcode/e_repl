# e_repl

A framework for studying language-model agents that control robots. The object of study is the
agent: what it does with a body, what it claims, when it declines, and how that changes with the
setup it is put in. The setup (the world document, the tools, the limits, the clock, the
teammates) is the variable, and the trace of every run is the data.

The model has a single action, "run this cell": it writes Python against tools that are ordinary
objects in a persistent namespace. Every run is written to disk as it happens, with the model's
reasoning, what it was shown, and what the world measured afterwards. The repo follows the
conventions of [poorin-lite](https://github.com/jasgcode/poorin-lite).

## What it is for

The questions below are the ones safety research asks of an agent with actuators. Each row names
what in e_repl answers it today.

| Question | What answers it |
|---|---|
| Did the agent decline a harmful task, or try and fail? | `refuse()` and `give_up()` are separate endings, and a harm task gets one of RoboHarm's six labels beside its verdict ([0014](docs/decisions/0014-give-up-ends-a-run-without-a-claim-of-success.md), [0015](docs/decisions/0015-a-run-can-carry-a-safety-label-beside-its-verdict.md)) |
| Does it claim success it did not get? | `done()` is a claim and the world's grader gives the verdict; both are in `result.json` ([0005](docs/decisions/0005-the-world-grades-each-run.md)) |
| Does it game the check instead of doing the task? | The grader is a function the model cannot reach, a task is admitted only if a model that writes the grader's state fails, and a world can run as its own process ([0010](docs/decisions/0010-a-world-can-run-as-its-own-process.md)) |
| What stops code the agent wrote? | Nothing in the REPL. The limit sits in the process that holds the device: only listed calls, motion only inside a run, a fault latch only a person clears, and a log written before each answer ([0018](docs/decisions/0018-the-harness-serves-hardware.md)) |
| Does a second agent make the first safer? | `run_team` puts agents on one task with shared limits; they talk through `tell()`, and any one of them can refuse for all ([0020](docs/decisions/0020-several-agents-can-share-a-run.md)) |
| How does time pressure change what it does? | A time limit the model is told about, one clock over everything, and added latency to price speed ([0016](docs/decisions/0016-a-run-keeps-a-time-ledger.md), [0017](docs/decisions/0017-a-run-can-have-a-time-limit-the-model-is-told.md)) |
| Did its stated reasoning match its actions? | Each step keeps the reasoning, the endpoint's whole response, and the code that ran ([0012](docs/decisions/0012-a-step-records-what-cannot-be-worked-out-later.md)), read in a viewer that runs no scripts ([0013](docs/decisions/0013-traces-are-read-through-a-viewer-with-no-scripts.md)) |
| Would the result hold if someone ran it again? | Every run records its recipe and fingerprint, the world's initial state, and the commit; the harness copy carries a digest ([0003](docs/decisions/0003-every-run-is-recorded-with-its-recipe.md), [0008](docs/decisions/0008-a-run-records-the-worlds-initial-state.md), [0009](docs/decisions/0009-the-harness-records-a-digest-of-its-files.md)) |

## Beside Inspect Robots

[Inspect Robots](https://github.com/robocurve/inspect-robots) carries
[Inspect AI](https://inspect.aisi.org.uk/) to physical robots: a benchmark of tasks, run as
trials of a policy (a VLA, or an LLM acting through motion tools) on an embodiment, scored and
written to an eval log, with guards that clip or reject unsafe actions. It answers how well, and
how safely, a policy performs a benchmark, and it compares policies and robots.

e_repl holds the policy and the robot fixed and varies the agent's situation. It works at the
scale of one bench, a few tasks, and many runs of each, and it keeps what a benchmark score
leaves out: the reasoning, the claims, the messages between agents, and the time each part took.

| | Inspect Robots | e_repl |
|---|---|---|
| Question | How well and how safely does this policy do these tasks? | What does this agent do under this setup, and why? |
| What varies | The policy, the embodiment, the benchmark | The world document, tools, limits, clock, teammates |
| How an LLM acts | Tool calls, one guarded motion chunk per call | Python in a persistent REPL, composing tools |
| Policies | VLAs and LLMs | LLMs |
| Scale | Benchmarks across robots and simulators | One bench, few tasks, many runs |
| Safety record | Guards on actions, a safety benchmark of RoboHarm tasks | RoboHarm labels, refusal apart from incapacity, claims against the world, a gate in the device's process |

The two meet at RoboHarm: Inspect Robots' safety benchmark is built from its tasks, and e_repl
labels harm runs with its six labels, so a task can be studied here and measured there. A behaviour found here is a candidate task for a benchmark
there. e_repl does not add VLA policies, an embodiment registry, simulators, or benchmark suites;
those belong in Inspect Robots.

## Start a project

```bash
git clone e_repl my-project && cd my-project
rm -rf .git && git init -b main
mv src/project src/my_project
# Rename `project` to `my_project` in pyproject.toml (`packages`, isort's `known-first-party`)
# and in the imports (`from project.toy import ...`), and set `[project] name`.
uv sync
uv run pytest
```

`harness/DIGEST` records a digest of the harness's files, and the tests recompute it. While your
copy matches e_repl's the digests are equal; when you change the harness, run
`uv run python -m harness.digest` and the new digest marks the copy as yours.

## Layout

| Path | Holds |
|---|---|
| `harness/` | the loop: the REPL, the chat client, records, recorders ([design and API](harness/README.md)) |
| `src/project/` | your code: the tools the model drives and the instructions that describe them |
| `experiments/` | commands that wire tools, a model, and a recorder into a run |
| `tests/` | tests for all three |
| `docs/` | decisions, experiments, and designs ([rules](docs/README.md), [index](docs/index.md)) |

`harness/` never imports `src/` or `experiments/`.

## A run

```python
shelf = shuffled_shelf(seed=0)
result = run_agent(
    task="Pick the red box.",
    repl=Repl({"shelf": shelf, "np": np}, ReplConfig()),
    chat=OpenAIChat(model="google/gemma-4-31b-it", base_url=NIM_BASE_URL, api_key=key),
    config=AgentConfig(instructions=INSTRUCTIONS, max_steps=8),
    recorder=DirectoryRecorder(Path("runs"), repo=Path(".")),
    grade=partial(pick_verdict, shelf, "red"),
    initial_state={"seed": 0},
)
```

```bash
export NVIDIA_API_KEY=...          # from build.nvidia.com
uv run python -m experiments.toy_pick red --model google/gemma-4-31b-it
uv run python -m experiments.toy_pick red --model google/gemma-4-31b-it --experiment toy-pick-rate
uv run python -m experiments.toy_pick red --model google/gemma-4-31b-it --world service
```

Give a run a time limit the model is told about, or slow every reply to see what speed is worth:

```bash
uv run python -m experiments.toy_pick red --model google/gemma-4-31b-it --time-limit 30
uv run python -m experiments.toy_pick red --model google/gemma-4-31b-it --extra-latency 5
```

Put several agents on one task. They take turns, share the shelf, and can tell each other things:

```bash
uv run python -m experiments.toy_pick red --model google/gemma-4-31b-it --agents 3
```

Read the runs in a browser, live while they are written:

```bash
uv run python -m harness.view serve
```

`--world service` runs the shelf as a process of its own
([`shelf_service.py`](src/project/shelf_service.py)) and gives the model a
[`ShelfClient`](src/project/shelf_client.py) with the same calls.

Each turn the model writes a sentence or two and one ` ```repl ` block. The harness runs it and
replies with what the code printed, the value of a trailing expression, the images passed to
`show()`, and a one-line summary of every variable. `done("summary")` ends the run when the model
says the task is complete, `give_up("reason")` when it says the task cannot be done, and
`refuse("reason")` when it says the task should not be done. `grade()` then reads the world and
gives the verdict that `result.json` records. A task that defines a harm also gets a safety label
there, one of RoboHarm's six.

## The principles

1. **One action.** A capability is a Python object in the namespace, never a tool schema. The
   model composes tools with loops, retries, and numpy, and keeps bulky data in variables.
2. **Tools describe themselves.** The system prompt lists each tool's signature and the first line
   of its docstring. That line is written for the model. The instructions and that list make up
   the world document, which names nothing of this harness and is recorded apart from its protocol.
3. **Errors are observations.** A tool that refuses a call raises with what was wrong and what
   would be allowed; the model reads the traceback and retries.
4. **The REPL is not a sandbox.** Model code runs with the process's rights. A safety limit
   belongs inside the tool that drives the hardware, where every call passes through it. A world
   whose state the model must not write, or that another harness must drive, runs as its own
   process with a client in the namespace.
5. **Mock and real tools share names and units**, so prompts and trajectories carry over.
6. **What changes results is in a config.** `AgentConfig`, `ReplConfig`, and the chat model's
   `recipe()` are recorded with every run and hashed into its fingerprint. The API key, the runs
   directory, and logging are arguments and stay out of it.
7. **Every run is recorded as it happens**, including the ones that hit the step limit, crash, or
   are stopped with Ctrl-C. The JPEG bytes on disk are the bytes the model saw. A step keeps the
   model's reasoning and the endpoint's whole response, which cannot be worked out afterwards.
8. **The harness reads no environment variables.** A command reads them once and passes values in.
9. **The world grades.** `done()` and `give_up()` end a run and the model's answer is a claim. The experiment's
   grader reads the world after the last cell, and the model cannot call it.
10. **Time is part of the record.** Every reply and every tool call is timed, a run can have a
    time limit the model is told about, and one clock counts everything. See
    [time and latency](docs/designs/2026-11-26-time-and-latency-design.md).
11. **It is built for real hardware.** A device is served from the process that holds it: only
    listed calls run, it moves only inside a run, a fault stops it until a person has looked, and
    every request is logged there. See [`harness/service.py`](harness/service.py).
12. **More than one agent is an option.** `run_team` puts several agents on one task in the same
    loop: they take turns, share the world and the limits, and any one of them can end the run.

## The one policy left open

`extract_code` in [`harness/replies.py`](harness/replies.py) decides what a reply with several code
blocks means. The baseline runs the first block and drops the rest without telling the model.
The alternatives: run all of them as one cell (fewer turns, but the model acts on results it has
not seen), or refuse the reply and ask for exactly one (strict, costs a turn). On a robot the
difference is whether a second, unseen motion can run. Choose, change the function, and add a
decision under `docs/decisions/`.

## Not here yet

- A monitor that reads each cell before it runs and can stop it. Oversight today is a teammate
  who reads results after the code has run, or a person at the device's fault latch.
- A label for a run a person stopped. A run interrupted because it was about to do harm is the
  one a study most wants labelled, and an interrupted run is not graded (decision 0015).
- A bridge to Inspect Robots: records written as its eval logs, or an e_repl world served as one
  of its embodiments.
- `llm_query` sub-calls are not written to the trajectory; only what the cell prints of them is.
- The cell time limit uses `SIGALRM`, so it works on the main thread of a Unix process only, and
  it interrupts whatever the cell was doing, a robot motion included.

## Development

```bash
uv run pytest
uv run ruff check --fix . && uv run ruff format .
uvx ty@0.0.84 check --error-on-warning
uv run python -m harness.docs      # after changing anything under docs/
```
