# e_repl

A framework for studying language-model agents that control robots. The agent is the object of
study: what it does, what it claims, when it declines. Its setup (instructions, tools, limits,
clock, teammates) is the variable, and the record of every run is the data.

The model has one action: run a cell of Python against tools in a persistent namespace. Code
follows [poorin-lite](https://github.com/jasgcode/poorin-lite)'s conventions.

## What it is for

| Question | What answers it |
|---|---|
| Did it decline a harmful task, or try and fail? | `refuse()` and `give_up()` are separate endings; harm tasks get a RoboHarm label ([0015](docs/decisions/0015-a-run-can-carry-a-safety-label-beside-its-verdict.md)) |
| Does it claim success it did not get? | `done()` is a claim; the world's grader gives the verdict ([0005](docs/decisions/0005-the-world-grades-each-run.md)) |
| Does it game the check? | The model cannot reach the grader, and the world can run in its own process ([0010](docs/decisions/0010-a-world-can-run-as-its-own-process.md)) |
| What stops the code it writes? | Not the REPL: the device's own process allows only listed calls and latches on a fault ([0018](docs/decisions/0018-the-harness-serves-hardware.md)) |
| Does a second agent make it safer? | `run_team` shares one task and its limits; any member can refuse for all ([0020](docs/decisions/0020-several-agents-can-share-a-run.md)) |
| How does time pressure change it? | A time limit the model is told, and added latency ([0017](docs/decisions/0017-a-run-can-have-a-time-limit-the-model-is-told.md)) |
| Did its reasoning match its actions? | Each step keeps the reasoning, the full response, and the code that ran ([0012](docs/decisions/0012-a-step-records-what-cannot-be-worked-out-later.md)) |
| Can someone rerun it? | Each run records its recipe, fingerprint, initial state, and commit ([0003](docs/decisions/0003-every-run-is-recorded-with-its-recipe.md)) |

## Beside Inspect Robots

[Inspect Robots](https://github.com/robocurve/inspect-robots) benchmarks policies (VLAs, or LLMs
acting through motion tools) across tasks and robots. e_repl holds the model and robot fixed and
varies the agent's setup, at the scale of one bench and many runs. Both use RoboHarm, so a
behaviour found here can become a benchmark task there. VLA policies, simulators, and benchmark
suites belong in Inspect Robots ([0021](docs/decisions/0021-e-repl-studies-agents-and-leaves-benchmarks-to-inspect-robots.md)).

| | Inspect Robots | e_repl |
|---|---|---|
| Asks | How well and how safely does this policy do these tasks? | What does this agent do under this setup, and why? |
| Varies | Policy, embodiment, benchmark | Instructions, tools, limits, clock, teammates |
| LLM acts by | One guarded motion per tool call | Python composing tools in a persistent REPL |

## Start

```bash
git clone e_repl my-project && cd my-project
rm -rf .git && git init -b main
mv src/project src/my_project   # then rename `project` in pyproject.toml and the imports
uv sync && uv run pytest
```

`harness/DIGEST` marks your copy of the harness. After changing it, run
`uv run python -m harness.digest`.

```python
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
export NVIDIA_API_KEY=...   # from build.nvidia.com
uv run python -m experiments.toy_pick red --model google/gemma-4-31b-it
#   --world service    the shelf in its own process, driven through a client
#   --time-limit 30    a limit the model is told about
#   --extra-latency 5  slow every reply
#   --agents 3         a team taking turns
uv run python -m harness.view serve   # read runs in a browser, live
```

Each turn the model writes one ` ```repl ` block. The harness replies with the output, the
trailing value, images passed to `show()`, and a summary of every variable. The run ends with
`done()`, `give_up()`, or `refuse()`, and the grader writes the verdict to `result.json`.

## Layout

| Path | Holds |
|---|---|
| `harness/` | the loop, REPL, chat client, and records ([API](harness/README.md)); never imports the rest |
| `src/project/` | the tools the model drives and the instructions that describe them |
| `experiments/` | commands that wire tools, a model, and a recorder into a run |
| `tests/` | tests for all three |
| `docs/` | decisions, experiments, designs ([rules](docs/README.md), [index](docs/index.md)) |

## The principles

1. **One action.** A capability is a Python object, never a tool schema.
2. **Tools describe themselves.** The prompt lists each signature and docstring's first line.
3. **Errors are observations.** A refused call raises with what was wrong and what is allowed.
4. **The REPL is not a sandbox.** Safety limits live in the tool that drives the hardware.
5. **Mock and real tools share names and units.**
6. **What changes results is in a config**, recorded and hashed into the run's fingerprint.
7. **Every run is recorded as it happens**, crashes and Ctrl-C included.
8. **The harness reads no environment variables.**
9. **The world grades**, after the last cell, with a grader the model cannot call.
10. **One clock times everything** ([design](docs/designs/2026-11-26-time-and-latency-design.md)).

## The one policy left open

`extract_code` in [`harness/replies.py`](harness/replies.py) runs the first code block of a reply
and drops the rest. Running them all, or refusing the reply, are the alternatives; on a robot the
choice decides whether an unseen motion can run. Choose, and record a decision.

## Not here yet

- A monitor that reads each cell before it runs and can stop it.
- A safety label for a run a person stopped.
- A bridge to Inspect Robots: its log format, or an e_repl world as its embodiment.
- `llm_query` sub-calls in the trajectory.
- A cell time limit that works off the main thread (it uses `SIGALRM`).

## Development

```bash
uv run pytest
uv run ruff check --fix . && uv run ruff format .
uvx ty@0.0.84 check --error-on-warning
uv run python -m harness.docs   # after changing docs/
```
