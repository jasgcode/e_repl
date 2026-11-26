# e_repl

A flexible framework/ harness for agentic robotics for studying how agents perform under different objective based setups. The model has a single action,
"run this cell"; tools are ordinary Python objects in the namespace; every run is written to disk
as a trajectory. The repo follows the conventions of [poorin-lite](https://github.com/jasgcode/poorin-lite).

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
