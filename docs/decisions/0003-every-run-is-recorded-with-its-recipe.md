---
date: 2026-11-17
status: adopted
summary: Every run streams to a recorder with its recipe, fingerprint, provenance, and outcome
---

# 0003. Every run streams to a recorder with its recipe, fingerprint, provenance, and outcome

## Decision

`run_agent` requires a `Recorder` and calls it at the start, after each step, and at the end. The
header holds the recipe (`AgentConfig`, `ReplConfig`, the chat model's `recipe()`), the system
prompt, and a fingerprint hashed from both. `DirectoryRecorder` adds the commit, the uncommitted
files, and the command line. A run ends with one of four outcomes, and a run that crashes or is
interrupted is recorded before the exception propagates.

## Why

- Successful runs are the data for filtering and fine-tuning, so a run must hold everything the
  model saw and wrote.
- A success rate compares runs of one setup. The fingerprint leaves out the task, so those runs
  share it; it includes the system prompt, so a changed tool docstring is a new setup.
- On a robot, Ctrl-C is how a person stops a bad run. That is a labelled failure worth keeping.
- Steps are appended as they finish, so a crash loses nothing that happened before it.
- With recording behind a protocol, tests use `MemoryRecorder` and the loop has no optional path.

## Alternatives

- An optional `log_dir`, as in `simple_101`: runs made without it leave nothing behind, and the
  loop carries a `None` check at each write.
- Returning the trajectory when the run ends: a crash at step 14 would lose steps 1 to 13.
- poorin-lite's artifact store (`name@version`, cached steps): an agent run on a live world is
  not a cacheable build.

## Cost if wrong

The format is JSON lines and JPEG files with no schema version. A change to the record needs a
one-off script over old run directories.
