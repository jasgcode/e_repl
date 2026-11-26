---
date: 2026-11-26
status: adopted
summary: Run directories are read through a local viewer that escapes all model text and carries no scripts
---

# 0013. Traces are read through a viewer with no scripts

## Decision

`harness.view` turns run directories into pages. `serve` answers on localhost from the
directories as they are at each request, so a run can be watched while it is written. `export`
writes one run as a single file with its images inside.

Every word that came from a model, a tool, or a task is escaped. No page carries a script, and
the served pages forbid them with a Content-Security-Policy. Nothing is written into a run
directory. Under a run directory the server gives out the three record files and the step images
and nothing else.

A run page puts what the model claimed beside what the world found, and the index shows one
success rate per fingerprint, counted as decision 0006 says.

## Why

- A trace is text a model chose. Read in a browser, unescaped, it is a page the model wrote. The
  people who read traces of a misbehaving model are the people this must not happen to.
- With no scripts allowed at all, an escaping mistake cannot run anything.
- Rendering from the directories on request means the page is never stale and the records stay
  the only copy. A run page reloads itself while its run has no result.
- The first question about a run is whether the model's account of it holds. A run it ended with
  `done()` that the world graded a failure is the one flag drawn as an alarm.
- An exported file can be attached to a report or sent to someone without the repository.

## Alternatives

- Static pages written beside each run: stale as soon as a step is added, and files in a record
  directory that are not part of the record.
- A notebook or a terminal printout: no images inline, and nothing to link to.
- Inspect's log format and its viewer: the standard in safety evaluation, with search and
  filtering this viewer lacks. It needs a dependency and a mapping from cells to tool calls. An
  export to that format can be added without changing a record.
- Filtering and search in the page with JavaScript: easier to use, and it would give up the rule
  that a page cannot run code.

## Cost if wrong

Without scripts there is no search box or live filter; the index filters by fingerprint through
links. A run with thousands of steps is one long page. The viewer reads every run directory on
each request to the index, which will be slow with thousands of runs.
