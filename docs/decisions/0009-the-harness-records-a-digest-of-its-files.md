---
date: 2026-11-21
status: adopted
summary: The harness records a digest of its own files, and the tests fail when a copy no longer matches it
---

# 0009. The harness records a digest of its own files

## Decision

`harness/DIGEST` holds `sha256:<hex>` over every file under `harness/` (relative path and bytes,
in path order), leaving out `DIGEST` itself, dot-files, and `__pycache__`.
`uv run python -m harness.digest` rewrites it, and the test suite runs `--check`.

## Why

- A project's `harness/` is a copy of this one. The first project that has one, simple_101, names
  the commit it copied in its `AGENTS.md`, and nothing checks that the files still match it.
- With the check, an edit to a copy fails that project's tests until the digest is regenerated.
  The changed `DIGEST` is then in the diff, so the copy leaves e_repl's on purpose or not at all.
- Two copies can be compared by one line each, with no second checkout and no network.
- Embodied First Exam copies its shared service layer into every task and fails CI when a copy
  differs from the source. See
  [the design](../designs/2026-11-21-embodied-first-exam-principles-design.md), principle 10.

## Alternatives

- Comparing against a checkout of e_repl in the tests: needs the network or a sibling directory,
  and no test here may need either.
- Installing the harness as a dependency: there would be no copy to drift. The template's premise
  is that a project owns its copy and may change it.
- Recording only the upstream commit, as simple_101 does: a statement that no program reads.
- Hashing only `*.py`: an edit to `README.md` or a new data file would pass unseen.

## Cost if wrong

Every change under `harness/` needs one more command, as a change under `docs/` does, and two
branches that both change the harness conflict on `DIGEST`. The check proves a copy equals what
was recorded, not that it equals e_repl's: someone can regenerate the digest without thinking.
