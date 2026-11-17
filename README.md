# e_repl

A template for agents that act through one persistent Python REPL. The model has a single action,
"run this cell"; tools are ordinary Python objects in the namespace; every run is written to disk
as a trajectory. It was pulled out of `simple_101` (an SO-101 arm driven by free NVIDIA NIM models)
and follows the conventions of [poorin-lite](https://github.com/jasgcode/poorin-lite). Copy it to
start a project; the copy is yours.

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
)
```

```bash
export NVIDIA_API_KEY=...          # from build.nvidia.com
uv run python -m experiments.toy_pick red --model google/gemma-4-31b-it
uv run python -m experiments.toy_pick red --model google/gemma-4-31b-it --experiment toy-pick-rate
```

Each turn the model writes a sentence or two and one ` ```repl ` block. The harness runs it and
replies with what the code printed, the value of a trailing expression, the images passed to
`show()`, and a one-line summary of every variable. `done("summary")` ends the run.

## The principles

1. **One action.** A capability is a Python object in the namespace, never a tool schema. The
   model composes tools with loops, retries, and numpy, and keeps bulky data in variables.
2. **Tools describe themselves.** The system prompt lists each tool's signature and the first line
   of its docstring. That line is written for the model.
3. **Errors are observations.** A tool that refuses a call raises with what was wrong and what
   would be allowed; the model reads the traceback and retries.
4. **The REPL is not a sandbox.** Model code runs with the process's rights. A safety limit
   belongs inside the tool that drives the hardware, where every call passes through it.
5. **Mock and real tools share names and units**, so prompts and trajectories carry over.
6. **What changes results is in a config.** `AgentConfig`, `ReplConfig`, and the chat model's
   `recipe()` are recorded with every run and hashed into its fingerprint. The API key, the runs
   directory, and logging are arguments and stay out of it.
7. **Every run is recorded as it happens**, including the ones that hit the step limit, crash, or
   are stopped with Ctrl-C. The JPEG bytes on disk are the bytes the model saw.
8. **The harness reads no environment variables.** A command reads them once and passes values in.

## The one policy left open

`extract_code` in [`harness/loop.py`](harness/loop.py) decides what a reply with several code
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
