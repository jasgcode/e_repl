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
- A tool that refuses a call raises an exception whose message says what was wrong and what would
  be allowed. Never return an error string or `False` for a refusal.
- The REPL is not a sandbox. Put a safety limit inside the tool that drives the hardware. Never
  weaken, bypass, or rewrite a safety gate the user owns; propose the change and stop.
- A mock tool and its real counterpart keep the same method names, argument order, and units.
- A value that changes what the model sees or does belongs in `AgentConfig`, `ReplConfig`, or a
  chat model's `recipe()`, so the run record and the fingerprint hold it. An argument is for how a
  run executes: the runs directory, the API key, logging.
- `harness/` reads no environment variables and holds no model IDs. A command in `experiments/`
  reads the environment once and passes values in.
- Every run goes through a `Recorder`. Never add a path through the loop that skips it.
- `harness/` must not import from `src/` or `experiments/`; project code imports `harness`, never
  the other way.

## Project docs

`docs/` holds the project's decisions, experiments, and designs; `docs/README.md` has the full
rules and a template for each kind.

- Read `docs/index.md` at the start of a task. Open only the files it points to.
- Add a decision (`docs/decisions/NNNN-<slug>.md`) when a change chooses between options, in the
  same commit as the change. Never rewrite a past decision; supersede it with a new one.
- Work on an experiment through its file. Open `docs/experiments/<slug>.md` before its first run,
  launch with `--experiment <slug>`, append dated entries to its `Log` as results arrive, and close
  it with a `Conclusion` that cites runs by directory name. A dead end closes too, saying why.
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
- No sleeps to wait for work; synchronize on events, barriers, or the result.
- Always fix tests you broke. Do not relax tolerances or hack around failures.
