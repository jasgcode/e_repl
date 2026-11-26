# harness

`harness` runs a chat model against a persistent Python namespace and records what happened. The
project's own code lives in `src/` and `experiments/` and imports `harness`; `harness` never
imports it.

## Modules

| Module | Holds |
|---|---|
| [`outcomes.py`](outcomes.py) | `CellStatus`, `Ending`, `Outcome`, `Verdict`, `Harm`, `HarmFinding`, `SafetyLabel`, `safety_verdict` |
| [`replies.py`](replies.py) | `extract_code`, `CODE_BLOCK`, `FinishReason`, `finish_reason`: what a reply says to the harness |
| [`images.py`](images.py) | `ImageSettings`, `ObservedImage`, `encode_image`: pixels to the JPEG bytes the model sees |
| [`repl.py`](repl.py) | `Repl`, `ReplConfig`, `Observation`, `ToolCall`, `describe_tools`, `public_methods` |
| [`chat.py`](chat.py) | the `ChatModel` protocol, `Reply`, `OpenAIChat`, `ScriptedChat`, `llm_query_tool` |
| [`records.py`](records.py) | `AgentPrompt`, `RunHeader` and its fingerprint, `Step`, `RunResult` |
| [`provenance.py`](provenance.py) | the commit, uncommitted files, and command line behind a run; `utc_timestamp` |
| [`runs.py`](runs.py) | run directories read back: `Run`, `read_runs`, `Standing`, `time_ledger`, `run_flags`, the rates |
| [`trajectory.py`](trajectory.py) | the `Recorder` protocol, `DirectoryRecorder`, `MemoryRecorder` |
| [`loop.py`](loop.py) | `AgentConfig`, `world_document`, `PROTOCOL`, `run_agent`, `run_team`, `Member`, `swarm` |
| [`localhost.py`](localhost.py) | `serving`: a server on localhost from a background thread, for the viewer and every service |
| [`docs.py`](docs.py) | the project docs' headers and the generated `docs/index.md` |
| [`digest.py`](digest.py) | the digest of these files that [`DIGEST`](DIGEST) records |
| [`view.py`](view.py) | run directories as pages: an index of runs and one page per run |
| [`service.py`](service.py) | a device held by one process and served to another: listed calls, a run, a fault latch, a log |
| [`library.py`](library.py) | `Library`, `snapshot`, `restore`: named texts an agent keeps from one round of a campaign to the next |
| [`campaign.py`](campaign.py) | `run_campaign`, the `Loop` and `Judge` protocols, `SelfJudge`, `CampaignConfig`, `Briefing`, `Round`, `judge_agreement`: rounds where a swappable judge keeps or reverts the agent's work and the loop grades it held out |
| [`operator.py`](operator.py) | `operator_verdict`: a person's yes, no, or skip, for a table no program can judge |

Imports run one way. Six modules need only the standard library: `outcomes`, `replies`,
`localhost`, `provenance`, `library`, and `runs` (which reads with the first two). `campaign` drives
runs and stands on `loop`'s records. The agent stack is
`images` ← `repl` ← `records` ← `trajectory` ← `loop`, with `chat` beside `repl`; it is the only
part that loads numpy, Pillow, or the chat client. `service`, `operator`, and `view` stand on the
light modules alone, so a hardware service, a grader, or a reader of runs loads none of the agent
stack; `tests/test_modules.py` checks it. `docs` and `digest` stand alone.

## The digest

A project's `harness/` is a copy of e_repl's. `DIGEST` holds `sha256:<hex>` over every file here
(relative path and bytes, in path order), leaving out `DIGEST` itself, dot-files, and
`__pycache__`. `uv run python -m harness.digest` rewrites it and `--check`, which the tests run,
fails when the files no longer match. An edit to a copy therefore fails that project's tests until
the digest is regenerated, and the changed `DIGEST` shows in the diff. Two copies with one digest
hold the same files.

## The REPL

`Repl(tools, config)` builds a namespace holding the tools and four builtins: `show(image,
caption="")`, `done(answer="")`, `give_up(reason)`, and `refuse(reason)`. A tool may not take one
of those names. The last three end the run: `done` when the model says the task is complete,
`give_up` when it says the task cannot be done, `refuse` when it says the task should not be
done. The last two need a reason.

`repl.run(code)` returns an `Observation`:

- `status`: `CellStatus.OK` or `CellStatus.ERROR`.
- `text`: stdout and stderr, then the `repr` of a trailing expression, then any traceback. The
  traceback leaves out the harness's own frames. Text longer than `max_output_chars` keeps its head
  and tail around a note that says how much was cut.
- `images`: what the cell passed to `show()`, at most `max_images_per_cell`.
- `ending`: `Ending.DONE`, `Ending.GAVE_UP`, or `Ending.REFUSED` when the cell called `done()`,
  `give_up()`, or `refuse()`, else `None`.
- `answer`: what the cell passed to that call, or `None`.
- `calls`: each call the cell made to a tool, with its name as the model wrote it
  (`robot.goto`), when it began on the run's clock, its seconds, and the type of what it raised. A tool's calls to itself are part
  of the outer call. The REPL times tools by replacing their public methods on the object, so a
  tool built on `__slots__` or a frozen dataclass is refused.

Every failure in the cell is an `ERROR` observation, including `SyntaxError` and a cell that runs
past `cell_timeout` seconds. `exit()` is ignored with a note. `KeyboardInterrupt` propagates.

`cell_timeout` is a backstop for runaway model code. It raises at whatever line is running, so a
tool caught mid-call reports nothing about where it stopped. A tool with long calls bounds them
itself, and `cell_timeout` stays longer than the longest of them.

`repl.variable_summary()` lists each variable the model defined, one line each; arrays appear as
shape and dtype.

## Images

`show()` takes an HxW gray or HxWx3 BGR `uint8` array. `encode_image` shrinks it to fit
`ImageSettings.max_side` and encodes it as JPEG once. Those bytes go to the model as a data URL
and to the run directory as a file.

## Chat models

`ChatModel` is a protocol with two methods: `complete(messages) -> Reply` and `recipe() -> dict`.
Messages are OpenAI chat dicts. A `Reply` holds the answer `text`, the model's `reasoning`, the
endpoint's `response` body verbatim, and the call's times: `request_seconds` for the request that
was answered, `seconds` for the whole call with its failed `attempts` and the waits between them.
Only `text` goes back into the context; the rest is for the record.

- `OpenAIChat(model, base_url, api_key, ...)` posts to `{base_url}/chat/completions`. It retries
  429 and 5xx responses, waiting `Retry-After` when sent. Thinking in `<think>` blocks or in the
  endpoint's reasoning field becomes the reply's `reasoning`. Its recipe holds the model, base
  URL, `max_tokens`, and `temperature`, and never the key.
  `NIM_BASE_URL` is NVIDIA's endpoint.
- `ScriptedChat(replies)` returns the replies in order and keeps every prompt in `.prompts`.
- `SlowedChat(chat, extra_seconds)` adds a fixed delay to every reply of another model and puts
  the delay in the recipe, to measure what latency costs a run.
- `llm_query_tool(models, default, image_settings)` returns an `llm_query(prompt, images=(),
  model=default)` function to put in the namespace, so the model can ask sub-models about images
  it holds in variables.

## The loop

`run_agent(task, repl, chat, config, recorder, grade, initial_state, judge_harm, experiment=None) -> RunResult`

1. Builds the system prompt from two parts joined by a blank line. `world_document(instructions,
   tools)` is the project's instructions and `describe_tools`: it names nothing of this harness,
   so another harness can be given the same text. `PROTOCOL` is the harness's own part: the cell
   format, `show`, `done`, and the step limit.
2. Hands the recorder a `RunHeader` holding both parts, the recipe, and `initial_state`: what the
   experiment says rebuilds the world as the run found it, such as a seed.
3. Each step: asks the model, runs the cell from `extract_code(reply)`, appends the observation,
   and hands the recorder a `Step`. An observation starts with `[step N of M status]`, so the model
   sees its budget. A reply with no code block gets an `ERROR` observation that restates the
   format. With `config.time_limit` set, the header also gives the seconds left, the protocol
   states the limit, and the run ends as `TIME_LIMIT` when it passes: no request is sent and no
   code is started after it, a late reply is recorded with status `not_run`, and a running cell
   is stopped.
4. Keeps images only in the latest `config.keep_images` observations that carry any; older ones
   become a placeholder. The recorder still receives every image.
5. Ends with an `Outcome`: `DONE`, `GAVE_UP`, `REFUSED`, `STEP_LIMIT`, `INTERRUPTED` (Ctrl-C), or
   `CRASHED` (an exception from outside the model's code). The last two are recorded and then
   re-raised.
6. After the last cell of a run that was not interrupted and did not crash, calls `grade()` once and stores its
   `Verdict(success, reason)` in the result. `grade` is the world's judgment of its own state:
   `done()` and `give_up()` end a run and decide nothing. A grader that cannot judge returns `None`. That run,
   an interrupted run, and a crashed run have no verdict, and a grader that raises makes the run
   `CRASHED`. Keep the grader out of the REPL's tools.

7. For a task that defines a harm, calls `judge_harm()` once at the same point. It returns a
   `HarmFinding` (`none`, `attempted`, or `crossed`, with a reason), and `safety_verdict` turns
   that, the run's ending, and the endpoint's content filter into one of RoboHarm's six labels,
   stored as the result's `safety`. Pass `judge_harm=None` for a task that defines no harm.

## Teams

`run_team(task, members, repl_config, config, recorder, grade, initial_state, judge_harm,
experiment=None) -> RunResult` runs two or more agents on one task, in the same loop.

```python
tools = {"robot": robot, "cam": cam}
members = [
    Member("mover", chat, tools, role="You move the arm."),
    Member("checker", chat, {"cam": cam}, role="You look after each move and say what you see."),
]
result = run_team(task, members, ReplConfig(), config, recorder, grade, initial_state, judge_harm)
```

- A `Member` is a name, a chat model, the tools it holds, and an optional role. `swarm(count,
  chat, tools)` makes equal members named `agent-1`, `agent-2`, and so on.
- Each member has its own conversation and its own namespace. Members that hold the same tool
  objects share a world; a member given fewer tools can do less.
- They take turns in the order given. `config.max_steps` and `config.time_limit` are the team's.
- `tell(message)` sends text to every other member. It appears at the top of the receiver's next
  observation under `[from your team]`.
- Any member's `done()`, `give_up()`, or `refuse()` ends the run for all. It is graded once.
- A step records its `agent`, what it `heard`, and what it `told`. `run.json` holds each member's
  prompt under `team`, and the recipe holds each member's name, role, and model.

A run with one agent is unchanged: the same prompt, the same fingerprint, no `tell`.

## Recipes and fingerprints

A run's recipe is `{"agent": AgentConfig, "repl": ReplConfig, "chat": chat.recipe()}`. Its
fingerprint is the first 8 hex digits of the SHA-256 of the recipe and the system prompt. The task
and the initial state are left out, so every run of one setup shares a fingerprint and the tools'
docstrings are part of it through the prompt.

## Run directories

`DirectoryRecorder(runs_dir, repo)` writes `runs_dir/<start time>-<fingerprint>/`:

| File | Holds |
|---|---|
| `run.json` | task, start time, initial state, experiment, fingerprint, recipe, system prompt and its two parts, provenance |
| `steps.jsonl` | per step: start time, seconds since the run began, seconds left of the limit, reasoning, reply, code, status, output, variables shown, image files and captions, answer, the model's latency, the wait on the endpoint, the cell's seconds, each tool call with its seconds, the endpoint's response |
| `stepNNN_I.jpg` | image `I` of step `NNN` |
| `result.json` | outcome, answer, step count, verdict (`success`, `reason`; `null` when not graded), safety (`label`, `reason`; `null` when the task defines no harm), finish time |

Steps are appended as they finish. `MemoryRecorder` keeps the same records in lists.

A run whose `result.json` has no verdict, or that has no `result.json`, is void and is left out of
a success rate; [`docs/README.md`](../docs/README.md) has the counting rule.

## Hardware

This harness is for real hardware. `service.py` holds a device in the process that owns it and
serves it to the agent's process:

```python
device = Device(robot, methods=("obs", "goto"), reads=("obs",), refusals=(MotionRejected,), reading=robot.obs)
service = Service(device, ActionLog(Path("outputs/actions/session.jsonl")))
with server(service, port=8101) as http_server:
    http_server.serve_forever()
```

- Only the listed `methods` run. A request for anything else is refused and logged.
- The device moves only inside a run. The experiment holds a `ServiceClient`, opens a run with
  the same time limit it gives the harness, and closes it before the grader looks:
  `with client.run(seconds): ...`. Outside a run and past its limit, only `reads` are answered.
- A call that raises anything but one of `refusals` is a fault. Every later motion is refused
  until a person presses Enter at the server (`service.clear_on_enter(sys.stdin)`).
- Every request is written to the `ActionLog` before it is answered: the call, its arguments, its
  outcome, the device's seconds for it, and `reading()` before and after.

`Service` holds these rules, and the HTTP handler only carries a request to it and its `Answer`
back, so a test or another transport drives the same rules.

The model's tool is a small class that wraps the `ServiceClient` and has the device's own
methods, so the model is shown those and nothing else. A tool object can set `call_facts` to a
dict during a call (a frame's age, the device's own time for a motion) and the step keeps it.

`operator_verdict(ask)` is the grader for a real table: yes, no, or skip, then a note.

## Reading runs

```bash
uv run python -m harness.view serve                              # http://127.0.0.1:8300
uv run python -m harness.view export runs/<run> --out trace.html # one file, images inside
```

`serve` reads an unfinished run again on each request, so a run page follows a run while it is
written; a run with a result never changes and is read once. `harness.runs` is the reader on its
own, for a script that counts rates or times runs without rendering a page.
The index lists runs newest first with one success rate per fingerprint. A run page shows what
the model claimed beside what the world found, with an alarm when the two disagree (`done` and a
failure, or `gave_up` and a success) or when the run crossed its task's harm threshold, then each step: the reasoning, the reply with the
code that ran and any that did not, what the model was shown, and the endpoint's response. Each
step has its own link (`#step-3`).

A run page also shows where the time went: one bar split between the model, the endpoint, the
tools, the model's own code, and the harness, then the model's median, 95th percentile, and
slowest latency, and each tool's calls and seconds. The index gives the median and slowest run of
each fingerprint.

All text from a model is escaped and no page carries a script, so a trace cannot act on the
browser that reads it. The viewer writes nothing into a run directory.
