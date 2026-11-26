---
date: 2026-11-21
status: adopted
summary: A run records what rebuilds the world's initial state, required and left out of the fingerprint
---

# 0008. A run records what rebuilds the world's initial state

## Decision

`run_agent` requires `initial_state`, a dict the experiment supplies: whatever rebuilds the world
as the run found it, such as a seed or a scene's name. It is a field of `RunHeader`, it is written
to `run.json`, and it is not part of the fingerprint. An experiment passes `{}` only when nothing
can rebuild its world.

## Why

- The toy's seed was recorded only inside the command line in the run's provenance. A run started
  from Python recorded no seed at all.
- With the task, the recipe, the commit, and the initial state in one file, a run can be set up
  again from `run.json` alone.
- A success rate pools runs of one setup across starting states, so the state stays out of the
  fingerprint, as the task does. Embodied First Exam does the same: its seed is recorded with each
  result and is not in the key that says which results are comparable.
- The argument is required so that leaving the state out is something an experiment writes down.

## Alternatives

- In the recipe: each seed would have its own fingerprint, and a rate over seeds would have no
  directory name to count by.
- Parsed from the provenance command line: depends on each command's option names, and is absent
  for a run that no command started.
- A typed record in place of a dict: what describes a starting state differs by project (a seed,
  a scene file, measured poses), as a recipe's chat settings do.

## Cost if wrong

A dict has no schema, so two experiments can describe one world in two ways. On hardware the
state a person arranged by hand is not something a dict can rebuild, and `{}` will be common.
