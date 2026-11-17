---
date: 2026-11-17
status: adopted
summary: Chat models sit behind a two-method protocol; messages stay OpenAI chat dicts
---

# 0002. Chat models sit behind a two-method protocol; messages stay OpenAI chat dicts

## Decision

The loop depends on `ChatModel`: `complete(messages) -> str` and `recipe() -> dict`. Messages are
OpenAI chat dicts end to end. `OpenAIChat` is the one real implementation: `requests` against any
`/chat/completions` endpoint, with the model, base URL, and API key passed to its constructor.
The harness holds no model IDs and reads no environment variables.

## Why

- NVIDIA NIM, vLLM, Ollama, and OpenRouter all speak this format, so one client covers the free
  hosted models and a local fine-tune.
- The messages are sent and logged verbatim. A message dataclass would be converted back at once.
- A model object carries its own settings, so `recipe()` can report exactly what produced a reply,
  and a role table ("planner", "ground") is the project's to define.
- Tests replace the model with `ScriptedChat` and need no network.

## Alternatives

- The `openai` package: one more dependency for one POST request.
- A provider-neutral message type: an abstraction with a single user.
- Role names and model IDs read from environment variables, as in `simple_101`: the run record
  could not say which model answered without also capturing the environment.

## Cost if wrong

A provider with a different wire format (Anthropic's, Gemini's) needs an adapter that translates
OpenAI chat dicts inside its `complete`. The loop and the recorders stay as they are.
