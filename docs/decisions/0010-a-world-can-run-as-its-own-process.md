---
date: 2026-11-21
status: adopted
summary: A world can run as its own process, with HTTP for the agent's client and a pipe for its owner's verdict
---

# 0010. A world can run as its own process

## Decision

A project may run its world as a service and put a client of it in the REPL. The toy shows the
shape once, beside the in-process shelf that stays the default:

- `project.shelf_service` holds the shelf and serves `GET /photo` and `POST /pick` on localhost.
  Every other path is 404.
- `project.shelf_client.ShelfClient` has the method names, argument order, units, and docstring
  first lines of `Shelf`. A refusal crosses the wire and is raised as the same `PickRefused`.
- The process that starts the service holds its pipes. The service prints its URL, ends the
  episode when its stdin closes, and prints the verdict. `ShelfService.finish` is the run's
  `grade`.

`toy_pick --world service` runs the experiment this way.

## Why

- Another harness can drive the same world by importing the client. The service has no code for
  any harness, which is how Embodied First Exam runs several of them on one task. See
  [the design](../designs/2026-11-21-embodied-first-exam-principles-design.md), principle 1.
- In process, a cell that assigns `shelf._picked` passes the grader, and `shelf.colors` gives the
  answer without a photo. The client holds a URL and no shelf, so neither exists to reach.
- The verdict has its own channel. No path returns it, so the model's code cannot ask the world
  whether it has succeeded.
- A pick is a call with an answer or a refusal, which is a request and a response. `requests` is
  already a dependency, and the server is the standard library's.
- The owner's channel is the child's pipes, so there is no token to keep from the model, no file
  to guard, and no port to probe for readiness: the URL is the first line the service prints.

## Alternatives

- Files on disk, as simple_101's camera server uses: right for a stream of frames, where a reader
  takes the newest. A pick needs its refusal back, and files would need a request and reply
  protocol built on top.
- A verdict path guarded by a token: the token would live in the harness's process, where a cell
  can find it, and the path would exist for any client to try.
- The service started on its own, as a hardware service is: it would outlive a crashed harness,
  which a robot wants. The toy ends with its owner so that tests and stopped runs leave no process
  behind.
- Replacing the in-process shelf: the template's first run should need one process and no port.

## Cost if wrong

A cell still runs with the owner's rights. It cannot change the shelf by assignment, but it can
attack the owner's process, so this is a boundary against shortcuts and not a sandbox. The
mock and real rule now covers three implementations, kept in step by one test of the tool list.
Each call costs a round trip. The client imports `PickRefused` from `project.toy`, so it is not
yet a file that stands alone.
