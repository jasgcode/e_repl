"""Chat models: the ``ChatModel`` protocol, an OpenAI-compatible HTTP client, and a scripted fake.

Messages stay in the OpenAI chat format (a list of ``{"role", "content"}`` dicts) because they are
sent and logged verbatim; converting them would add a layer that every provider undoes.
"""

import dataclasses
import re
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import requests

from harness.images import ImageSettings, ObservedImage, encode_image
from harness.replies import finish_reason

type Message = dict[str, Any]

NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"
"""NVIDIA's hosted NIM endpoints: OpenAI-compatible, with a rate-limited free tier."""

_THINK_BLOCK = re.compile(r"<think>(.*?)</think>\s*", re.DOTALL)
_THINK_OPEN = "<think>"
_REASONING_FIELDS = ("reasoning_content", "reasoning")
"""Where OpenAI-compatible endpoints put thinking they return apart from the content."""
_RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
_MAX_BACKOFF = 60.0
_ERROR_BODY_CHARS = 500


class ChatError(RuntimeError):
    """The endpoint refused the request, or kept failing after every retry."""


@dataclass(frozen=True)
class Reply:
    """One answer from a chat model, with everything the endpoint said around it."""

    text: str
    """The answer with thinking removed: what the loop parses and what goes back into the context."""
    reasoning: str | None = None
    """The model's thinking, from ``<think>`` blocks and the endpoint's reasoning field. Recorded, never sent back."""
    response: dict[str, Any] | None = None
    """The endpoint's response body, verbatim: the served model, the finish reason, the token usage."""
    seconds: float = 0.0
    """Wall time of the call, with retries and their waits."""
    attempts: int = 1
    request_seconds: float = 0.0
    """Wall time of the one request that was answered: the model's latency, without earlier
    attempts or the waits between them."""

    @property
    def finish_reason(self) -> str | None:
        """Why the endpoint says the reply stopped (a ``FinishReason``), if it said."""
        return finish_reason(self.response)


class ChatModel(Protocol):
    def complete(self, messages: Sequence[Message]) -> Reply:
        """The model's reply to ``messages``."""
        ...

    def recipe(self) -> dict[str, Any]:
        """The settings that change replies, for the run record. Never secrets."""
        ...


def image_part(image: ObservedImage) -> Message:
    return {"type": "image_url", "image_url": {"url": image.data_url()}}


def split_thinking(content: str) -> tuple[str, str | None]:
    """The answer in ``content`` and the thinking in its ``<think>`` blocks, or ``None`` without any.

    A ``<think>`` that never closes is a reply cut off mid-thought: everything after it is thinking,
    so code inside it is not mistaken for the answer and run.
    """
    thoughts = _THINK_BLOCK.findall(content)
    answer, unclosed, cut_off = _THINK_BLOCK.sub("", content).partition(_THINK_OPEN)
    if unclosed:
        thoughts.append(cut_off)
    thinking = "\n\n".join(thought.strip() for thought in thoughts if thought.strip())
    return answer.strip(), thinking or None


@dataclass(frozen=True)
class OpenAIChat:
    """A model behind an OpenAI-compatible ``/chat/completions`` endpoint (NIM, vLLM, OpenRouter, ...).

    Rate limits (429) and server errors are retried, waiting ``Retry-After`` when the endpoint sends
    it and backing off exponentially otherwise.
    """

    model: str
    base_url: str
    api_key: str = field(repr=False)
    max_tokens: int = 2048
    temperature: float = 0.2
    retries: int = 6
    timeout: float = 300.0
    session: requests.Session = field(default_factory=requests.Session, repr=False, compare=False)

    def recipe(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "base_url": self.base_url,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

    def complete(self, messages: Sequence[Message]) -> Reply:
        payload = {
            "model": self.model,
            "messages": list(messages),
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Accept": "application/json"}
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        start = time.monotonic()
        for attempt in range(self.retries):
            sent = time.monotonic()
            response = self.session.post(url, json=payload, headers=headers, timeout=self.timeout)
            if response.status_code in _RETRY_STATUSES:
                time.sleep(_retry_wait(response, attempt))
                continue
            if response.status_code >= 400:
                body = response.text[:_ERROR_BODY_CHARS]
                raise ChatError(f"{self.model}: HTTP {response.status_code}: {body}")
            body = response.json()
            message = body["choices"][0]["message"]
            text, thinking = split_thinking(message.get("content") or "")
            apart = [message[name] for name in _REASONING_FIELDS if message.get(name)]
            reasoning = "\n\n".join([*apart, *([thinking] if thinking else [])])
            now = time.monotonic()
            return Reply(text, reasoning or None, body, now - start, attempt + 1, now - sent)
        raise ChatError(f"{self.model}: still rate-limited or failing after {self.retries} tries")


def _retry_wait(response: requests.Response, attempt: int) -> float:
    retry_after = response.headers.get("Retry-After")
    if retry_after is not None:
        return float(retry_after)
    return min(_MAX_BACKOFF, 2.0 ** (attempt + 1))


@dataclass(frozen=True)
class SlowedChat:
    """A chat model with a fixed delay added to every call, to measure what latency costs a run.

    The delay is in the recipe, so runs at different delays have different fingerprints.
    """

    chat: ChatModel
    extra_seconds: float
    sleep: Callable[[float], None] = field(default=time.sleep, repr=False, compare=False)

    def recipe(self) -> dict[str, Any]:
        return {**self.chat.recipe(), "extra_seconds": self.extra_seconds}

    def complete(self, messages: Sequence[Message]) -> Reply:
        self.sleep(self.extra_seconds)
        reply = self.chat.complete(messages)
        return dataclasses.replace(
            reply,
            seconds=reply.seconds + self.extra_seconds,
            request_seconds=reply.request_seconds + self.extra_seconds,
        )


@dataclass
class ScriptedChat:
    """Replies from a fixed script, for tests and dry runs. Keeps every prompt it was sent."""

    replies: Iterable[str]
    prompts: list[list[Message]] = field(default_factory=list)
    _replies: Iterator[str] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._replies = iter(self.replies)

    def recipe(self) -> dict[str, Any]:
        return {"model": "scripted"}

    def complete(self, messages: Sequence[Message]) -> Reply:
        self.prompts.append([dict(message) for message in messages])
        reply = next(self._replies, None)
        if reply is None:
            raise ChatError(f"script ran out after {len(self.prompts) - 1} replies")
        return Reply(reply)


def llm_query_tool(models: Mapping[str, ChatModel], default: str, image_settings: ImageSettings) -> Callable[..., str]:
    """An ``llm_query`` function for the REPL: one question to a sub-model, optionally about images.

    The model reaches every entry of ``models`` by name; ``default`` is the one it gets unasked.
    """
    if default not in models:
        raise ValueError(f"default model {default!r} is not one of {sorted(models)}")

    def llm_query(prompt: str, images: Sequence[np.ndarray] = (), model: str = default) -> str:
        if model not in models:
            raise ValueError(f"unknown model {model!r}: use one of {sorted(models)}")
        content: list[Message] = [image_part(encode_image(np.asarray(i), "", image_settings)) for i in images]
        content.append({"type": "text", "text": prompt})
        return models[model].complete([{"role": "user", "content": content}]).text

    llm_query.__doc__ = (
        f"Ask a sub-model one question, optionally about uint8 images; returns its reply. "
        f"model: one of {', '.join(sorted(models))}."
    )
    return llm_query
