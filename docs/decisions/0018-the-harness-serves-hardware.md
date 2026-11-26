---
date: 2026-11-26
status: adopted
summary: The harness ships the service a hardware project needs, with listed calls, a run, a fault latch, and a log
---

# 0018. The harness serves hardware

## Decision

`harness.service` holds a device in one process and serves its listed calls to another. A project
says what the device is with a `Device`: the object, the calls a client may make, which of them
only read, which exceptions are refusals, and how to read the device's state. The module does the
rest, the same way for every device:

- Only listed calls run. Anything else is refused and logged as it was asked.
- The device moves only inside a run. `ServiceClient.open_run(seconds)` starts one and
  `close_run()` ends it. Outside a run, and once its seconds have passed, every call that is not
  a reading is refused.
- A call that raises anything but a listed refusal trips a latch, and nothing moves until a
  person presses Enter at the server.
- Every request is logged, before it is answered, by the process that holds the device.

`harness.operator.operator_verdict` is the grader for a table no program can judge: yes, no, or
skip, then a note.

## Why

- This harness is for real hardware. The parts every hardware project needs were written once, in
  the first project that had an arm, and would have been written again in the next.
- They are the parts where a mistake moves something. One tested module is safer than one copy
  per robot.
- The run is how a time limit holds where the hardware is. The harness stops starting code at the
  limit, and a thread a model left behind, or a second client, is not the harness. The device
  refusing to move is the limit that cannot be skipped.
- Closing the run before the grader looks makes the table still while it is judged, which
  decision 0005 assumed and nothing enforced.
- The project keeps what is its own: the list of calls, what counts as a refusal, and the
  device's safety gate, which the module never touches.

## Alternatives

- Leaving it in each project: no shared tests, and each project free to forget the latch.
- A device that moves whenever asked, with the limit only in the harness: enough for an honest
  model, and the harness is the process the model's code runs in.
- Stopping a motion at the limit: a motion cut partway leaves the device somewhere nobody chose.
  A call that has started is finished; no new one starts.
- Serving arrays as well as numbers: a camera would then need no files. JSON carries a call's
  arguments and results, and the toy shelf, which sends a photo, keeps its own small server.

## Cost if wrong

The model's code runs with the operator's rights and can open a run itself; each opening is
logged, so two for one run show it, and only another operating-system user prevents it. An
experiment that forgets to open a run gets a device that refuses to move. There are now two
servers in this repository, this one and the toy shelf's. None of it has run on a robot: the
tests use a fake device.
