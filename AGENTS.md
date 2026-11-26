# Agent Guidelines

This repository is a project started from e_repl, a template for agents that act through one
persistent Python REPL. `README.md` states the principles; `harness/README.md` describes the
design and the API. The code conventions below are poorin-lite's.

## Development

```bash
# Tests (CPU, no network, no API key)
uv run pytest

# Lint and format
uv run ruff check --fix . && uv run ruff format .

# Type checking: errors and warnings fail
uvx ty@0.0.84 check --error-on-warning

# Regenerate docs/index.md after changing anything under docs/ (the tests check it)
uv run python -m harness.docs

# Regenerate harness/DIGEST after changing anything under harness/ (the tests check it)
uv run python -m harness.digest
```

- Python >= 3.13.
- Dependencies may be upgraded freely. Each dependency change is its own commit, and its
  `uv.lock` diff is checked for versions that moved without being asked.
- Commit before a run you will cite: `run.json` records the commit and lists uncommitted files,
  and a run from a dirty tree cannot be rebuilt.

## The harness's rules

- The model has one action: run a cell. Add a capability as a Python object in the namespace.
  Never add a tool schema, a second action type, or a parser for model output beyond
  `extract_code`.
- A tool's signature and the first line of its docstring are what the model sees. Write that line
  for the model: what the call does, its units, its frame.
- `AgentConfig.instructions` describes the world: what is in it, the units, the rules. Nothing in
  it names the harness (cells, `show`, `done`, steps); that text belongs in `PROTOCOL`.
- An experiment passes `run_agent` the `initial_state` that rebuilds its world (a seed, a scene's
  name), and `{}` only when nothing can.
- A tool that refuses a call raises an exception whose message says what was wrong and what would
  be allowed. Never return an error string or `False` for a refusal. A refused call does nothing.
- A tool never changes an argument silently. It refuses the call, or it carries out the adjusted
  call and its result says what was adjusted.
- A tool whose call can stop partway (a motion, a timed wait) bounds the call itself and returns
  or raises with how much ran and the state it stopped in. `ReplConfig.cell_timeout` is a backstop
  for runaway model code: it interrupts a tool at whatever line is running and the tool reports
  nothing, so set it longer than the longest call any tool allows.
- A tool returns data and never calls `show()`. What enters the context is the decision of the
  model's code.
- The instructions say what the hardware does between cells (holds its pose, goes limp) and what
  a new call does to a motion still under way.
- The REPL is not a sandbox. Put a safety limit inside the tool that drives the hardware. Never
  weaken, bypass, or rewrite a safety gate the user owns; propose the change and stop.
- A mock tool and its real counterpart keep the same method names, argument order, and units.
  So does a client of a world that runs as its own process, down to the first line of each
  docstring; a test compares what `describe_tools` shows for the two.
- A world that runs as a service gives its verdict to the process that started it, never over
  the channel its client uses, and has no path that returns state the task asks the model to
  find. Its safety limit is on the service's side, in the class that drives the hardware.
- A value that changes what the model sees or does belongs in `AgentConfig`, `ReplConfig`, or a
  chat model's `recipe()`, so the run record and the fingerprint hold it. An argument is for how a
  run executes: the runs directory, the API key, logging.
- `harness/` reads no environment variables and holds no model IDs. A command in `experiments/`
  reads the environment once and passes values in.
- Every run goes through a `Recorder`. Never add a path through the loop that skips it.
- A record keeps everything the model wrote, thought, and was shown. Never drop or shorten text
  before it is recorded; shorten it for the model, and record what was sent and what came back.
- Hardware is served with `harness.service`. Never add a call, a flag, or a timeout that clears
  the fault latch, never serve a call that cuts power or parks the device, and never answer a
  request without a line in the `ActionLog`.
- A device moves only inside a run. An experiment on hardware opens the run with its time limit
  and closes it before the grader looks. Opening and closing a run are never on a tool the model
  holds.
- A time is measured on hardware. A mock checks the code; a time from a mock is never reported as
  the task's.
- A run has one clock, and everything counts on it. Never add a second clock that leaves out the
  endpoint, the tools, or the harness, and never run code after `time_limit` has passed.
- A setting that makes a run faster (a model option, an image size, a delay) goes in the recipe,
  so the fingerprint holds it.
- Report a time as the median and the slowest over every graded run of one fingerprint, with its
  limit, never as the best run.
- `harness.view` shows records to people. Escape every value it puts in a page and never add a
  script to one: a trace is text a model chose.
- A run's success is the `Verdict` of the experiment's grader, never `Outcome.DONE` or the model's
  answer. `give_up(reason)` is how a model stops without claiming success and `refuse(reason)` how
  it declines a task; never fold the endings back into one.
- A task that defines a harm passes `judge_harm`, which reports what the world observed (`none`,
  `attempted`, `crossed`). The label comes from `safety_verdict` alone: never compute one of the
  six labels in a task, and never count a capability failure as safe. A grader is a function the experiment holds: never a tool, and never a public method of
  one, because the model is shown those and could call it. A grader that cannot judge returns
  `None`; it never guesses.
- `harness/` must not import from `src/` or `experiments/`; project code imports `harness`, never
  the other way.
- `harness/DIGEST` identifies this copy of the harness. It equals e_repl's at some commit, or the
  harness was changed here on purpose, and the commit that changes the digest says which.

## Project docs

`docs/` holds the project's decisions, experiments, and designs; `docs/README.md` has the full
rules and a template for each kind.

- Read `docs/index.md` at the start of a task. Open only the files it points to.
- Add a decision (`docs/decisions/NNNN-<slug>.md`) when a change chooses between options, in the
  same commit as the change. Never rewrite a past decision; supersede it with a new one.
- Work on an experiment through its file. Open `docs/experiments/<slug>.md` before its first run,
  launch with `--experiment <slug>`, append dated entries to its `Log` as results arrive, and close
  it with a `Conclusion` that cites runs by directory name. A dead end closes too, saying why.
- A success rate counts graded runs only. Write the number of void runs (crashed, interrupted, or
  with no `result.json`) beside it, and never repeat a graded run to replace its result.
- Write specs and implementation plans to `docs/designs/` (`YYYY-MM-DD-<slug>-design.md`,
  `YYYY-MM-DD-<slug>-plan.md`), not to the repository root or `docs/superpowers/`.
- Run `uv run python -m harness.docs` before committing a change under `docs/`.

## Communication & Commits

- NEVER SAY "You're absolutely right!"
- Commit only when asked. Never push unless asked.
- Commit titles are imperative, at most 72 characters, with a `[scope]` prefix such as
  `[harness]` or `[project]`. The body states what changed and why.
- NEVER credit yourself in commit messages: no `Co-Authored-By` trailer, no "Generated with …"
  line, no emoji attribution, even if a tool default suggests one.

## Code Style

- All imports at the top of the file. No local imports except to break circular dependencies or guard optional deps. No `TYPE_CHECKING` guards — fix cycles structurally via protocols.
- Prefer top-level functions over classes when code does not mutate shared state. Reduce deep inheritance hierarchies.
- Use early returns to reduce nesting.
- Document public APIs with concise Google-style docstrings. Skip docstrings on trivial functions with clear names.
- Prefer `dataclasses.replace` over mutating config arguments in-place.
- Prefer logging over `print` (except in scripts and debugging).
- Resolve environment-dependent defaults once and fail fast on unknown inputs.
- No ad-hoc compatibility hacks (`hasattr(m, "old_attr")`); update code consistently.
- Prefer small concrete helpers over abstraction that adds indirection without reuse. Start simple; abstract only under real pressure.
- Delete dead code: unused parameters, stale options, old experiments.
- Top-level constants for magic strings/numbers.
- Separate computation from I/O (split compute from upload/write).
- Use context managers for resource lifecycle.

## Naming

- No `*_utils.py` — use descriptive names like `text_cleaning.py`.
- Function names should reflect return types (`probe_task` → `task_status`).
- No `_s` suffix for seconds (assumed in this codebase). No abbreviations like `exe` — use `exec` or full words.

## Types & Data Structures

- Dataclass/namedtuple over raw dicts. `StrEnum` over string keys.
- Use `Protocol` for decoupling; avoid hard-coupling to concrete types.
- Avoid `X | str` unions that require `isinstance` checks — pick one input type.
- Replace compound booleans encoding state with an enum.
- Chat messages are the exception: they stay OpenAI chat dicts, because they are sent and logged
  verbatim (decision 0002).

## Configuration

- No `default_*` wrappers that obscure underlying mechanisms.
- Force explicit specification of critical parameters (no silent defaults).
- Centralize defaults in one canonical location.
- Prefer explicit constructor/config parameters over env vars.
- Composition over inheritance: embed sub-configs, don't subclass.

## API Design

- Accept only what's necessary. Replace boolean flags with meaningful parameters (e.g., `num_workers: int` instead of `parallel: bool`).
- Use separate classes over boolean flags for variant behavior.
- Normalize inputs to a standard format once at the boundary, not throughout.

## Error Handling

- Let exceptions propagate by default.
- Only catch to add meaningful context and re-raise, or to intentionally alter control flow.
- NEVER swallow exceptions unless specifically requested.
- Assert liberally; prefer `raise ValueError` over silent fallbacks.
- `Repl.run` is the one place that turns exceptions into values: a failure in the model's code is
  an observation. Do not copy that pattern anywhere else.

## Deprecation

**NO BACKWARD COMPATIBILITY**: Update all call sites instead. Only add compatibility shims if the user explicitly requests it.

## Comments

- Write comments for module/class-level behavior or subtle logic. Do not restate the code.
- Delete stale comments immediately on discovery.
- Inline comments to clarify non-obvious boolean arguments.

## LLM-Generated Code Pitfalls

Watch for and eliminate these patterns in generated code:
- Over-protective try/except and defensive None checks
- Tautological and guard-mirroring tests (type exists, constant has value, nonpositive scalar
  raises without a regression or public compatibility contract)
- Verbose/redundant docstrings and `__all__` in `__init__.py`
- Boolean dispatch instead of separate classes
- Environment variables instead of explicit parameters

## Code Reuse

Before writing any utility function, helper, or data structure, search `harness/` and
`src/project/`, then `pyproject.toml` for available third-party packages. If a suitable
implementation exists, use it. Do not create parallel implementations.

## Testing

- Test behavior, not implementation: a test names the change that would make it fail.
- No tautological tests that restate a constant or a type.
- Prefer real objects and small fakes over mocks; never assert on a mock's own behavior. Use
  `ScriptedChat` for the model and `MemoryRecorder` for the recorder.
- No test needs the network, an API key, or hardware.
- A task is admitted by three scripted models run through `run_agent` with its grader: a reference
  solution graded a success, a model that only calls `done()` graded a failure, and a model that
  writes the state the grader reads graded a failure.
- No sleeps to wait for work; synchronize on events, barriers, or the result.
- Always fix tests you broke. Do not relax tolerances or hack around failures.
