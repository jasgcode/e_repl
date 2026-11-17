# harness

`harness` runs a chat model against a persistent Python namespace and records what happened. The
project's own code lives in `src/` and `experiments/` and imports `harness`; `harness` never
imports it.

## Modules

| Module | Holds |
|---|---|
| [`images.py`](images.py) | `ImageSettings`, `ObservedImage`, `encode_image`: pixels to the JPEG bytes the model sees |
| [`repl.py`](repl.py) | `Repl`, `ReplConfig`, `Observation`, `CellStatus`, `describe_tools` |
| [`chat.py`](chat.py) | the `ChatModel` protocol, `OpenAIChat`, `ScriptedChat`, `llm_query_tool` |
| [`records.py`](records.py) | `RunHeader` and its fingerprint, `Step`, `RunResult`, `Outcome` |
| [`provenance.py`](provenance.py) | the commit, uncommitted files, and command line behind a run |
| [`trajectory.py`](trajectory.py) | the `Recorder` protocol, `DirectoryRecorder`, `MemoryRecorder` |
| [`loop.py`](loop.py) | `AgentConfig`, `system_prompt`, `extract_code`, `run_agent` |
| [`docs.py`](docs.py) | the project docs' headers and the generated `docs/index.md` |

Imports run one way: `images` ← `repl` ← `records` ← `trajectory` ← `loop`, with `chat` beside
`repl`. `docs` stands alone.

## The REPL

`Repl(tools, config)` builds a namespace holding the tools and two builtins, `show(image,
caption="")` and `done(answer="")`. A tool may not be named `show` or `done`.

`repl.run(code)` returns an `Observation`:

- `status`: `CellStatus.OK` or `CellStatus.ERROR`.
- `text`: stdout and stderr, then the `repr` of a trailing expression, then any traceback. The
  traceback leaves out the harness's own frames. Text longer than `max_output_chars` keeps its head
  and tail around a note that says how much was cut.
- `images`: what the cell passed to `show()`, at most `max_images_per_cell`.
- `answer`: what the cell passed to `done()`, or `None`. `observation.done` is `answer is not None`.

Every failure in the cell is an `ERROR` observation, including `SyntaxError` and a cell that runs
past `cell_timeout` seconds. `exit()` is ignored with a note. `KeyboardInterrupt` propagates.

`repl.variable_summary()` lists each variable the model defined, one line each; arrays appear as
shape and dtype.

## Images

`show()` takes an HxW gray or HxWx3 BGR `uint8` array. `encode_image` shrinks it to fit
`ImageSettings.max_side` and encodes it as JPEG once. Those bytes go to the model as a data URL
and to the run directory as a file.

## Chat models

`ChatModel` is a protocol with two methods: `complete(messages) -> str` and `recipe() -> dict`.
Messages are OpenAI chat dicts.

- `OpenAIChat(model, base_url, api_key, ...)` posts to `{base_url}/chat/completions`. It retries
  429 and 5xx responses, waiting `Retry-After` when sent, and strips `<think>` blocks. Its recipe
  holds the model, base URL, `max_tokens`, and `temperature`, and never the key.
  `NIM_BASE_URL` is NVIDIA's endpoint.
- `ScriptedChat(replies)` returns the replies in order and keeps every prompt in `.prompts`.
- `llm_query_tool(models, default, image_settings)` returns an `llm_query(prompt, images=(),
  model=default)` function to put in the namespace, so the model can ask sub-models about images
  it holds in variables.

## The loop

`run_agent(task, repl, chat, config, recorder, experiment=None) -> RunResult`

1. Builds the system prompt: `config.instructions`, then the protocol, then `describe_tools`.
2. Hands the recorder a `RunHeader`.
3. Each step: asks the model, runs the cell from `extract_code(reply)`, appends the observation,
   and hands the recorder a `Step`. A reply with no code block gets an `ERROR` observation that
   restates the format.
4. Keeps images only in the latest `config.keep_images` observations that carry any; older ones
   become a placeholder. The recorder still receives every image.
5. Ends with an `Outcome`: `DONE`, `STEP_LIMIT`, `INTERRUPTED` (Ctrl-C), or `CRASHED` (an exception
   from outside the model's code). The last two are recorded and then re-raised.

## Recipes and fingerprints

A run's recipe is `{"agent": AgentConfig, "repl": ReplConfig, "chat": chat.recipe()}`. Its
fingerprint is the first 8 hex digits of the SHA-256 of the recipe and the system prompt. The task
is left out, so every run of one setup shares a fingerprint and the tools' docstrings are part of
it through the prompt.

## Run directories

`DirectoryRecorder(runs_dir, repo)` writes `runs_dir/<start time>-<fingerprint>/`:

| File | Holds |
|---|---|
| `run.json` | task, experiment, fingerprint, recipe, system prompt, provenance |
| `steps.jsonl` | per step: reply, code, status, output, image files and captions, answer |
| `stepNNN_I.jpg` | image `I` of step `NNN` |
| `result.json` | outcome, answer, step count |

Steps are appended as they finish. `MemoryRecorder` keeps the same records in lists.
