"""The OpenAI-compatible client against a fake endpoint, and the llm_query tool."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest

from harness import chat as chat_module
from harness.chat import ChatError, OpenAIChat, Reply, ScriptedChat, SlowedChat, llm_query_tool, split_thinking
from harness.images import ImageSettings


@dataclass
class FakeResponse:
    status_code: int
    body: dict[str, Any] = field(default_factory=dict)
    headers: dict[str, str] = field(default_factory=dict)
    text: str = ""

    def json(self) -> dict[str, Any]:
        return self.body


@dataclass
class FakeEndpoint:
    """Stands in for ``requests.Session``: answers each POST with the next response."""

    responses: list[FakeResponse]
    requests: list[dict[str, Any]] = field(default_factory=list)

    def post(self, url: str, **kwargs: Any) -> FakeResponse:
        self.requests.append({"url": url, **kwargs})
        return self.responses[len(self.requests) - 1]


def _reply(content: str) -> FakeResponse:
    return FakeResponse(200, {"choices": [{"message": {"content": content}}]})


def _rate_limited() -> FakeResponse:
    return FakeResponse(429, headers={"Retry-After": "0"})


def _chat(endpoint: FakeEndpoint, retries: int = 3) -> OpenAIChat:
    return OpenAIChat(model="m", base_url="http://host/v1/", api_key="secret", retries=retries, session=endpoint)


def test_a_rate_limited_request_is_retried_and_the_thinking_is_kept_apart_from_the_answer():
    endpoint = FakeEndpoint([_rate_limited(), _reply("<think>hmm</think>\nthe answer")])
    reply = _chat(endpoint).complete([{"role": "user", "content": "q"}])
    assert reply.text == "the answer" and reply.reasoning == "hmm" and reply.attempts == 2
    assert [r["url"] for r in endpoint.requests] == ["http://host/v1/chat/completions"] * 2
    assert endpoint.requests[0]["headers"]["Authorization"] == "Bearer secret"


def test_a_reply_keeps_the_endpoints_whole_response_and_its_separate_reasoning_field():
    body = {
        "model": "m-2026-10",
        "choices": [{"finish_reason": "length", "message": {"content": "cut", "reasoning_content": "first I"}}],
        "usage": {"prompt_tokens": 7, "completion_tokens": 3},
    }
    reply = _chat(FakeEndpoint([FakeResponse(200, body)])).complete([])
    assert reply.text == "cut" and reply.reasoning == "first I"
    assert reply.response == body


def test_a_thought_that_was_cut_off_is_all_thinking_so_code_inside_it_is_not_the_answer():
    answer, thinking = split_thinking("ok <think>plan:\n```repl\nrobot.release()\n```")
    assert answer == "ok" and thinking is not None and "robot.release()" in thinking
    assert split_thinking("just the answer") == ("just the answer", None)


def test_a_models_latency_is_kept_apart_from_the_waits_on_a_rate_limited_endpoint(monkeypatch):
    class Clock:
        now = 0.0

        def monotonic(self) -> float:
            return self.now

        def sleep(self, seconds: float) -> None:
            self.now += seconds

    clock = Clock()
    monkeypatch.setattr(chat_module, "time", clock)

    class TwoSecondEndpoint(FakeEndpoint):
        def post(self, url: str, **kwargs: Any) -> FakeResponse:
            clock.now += 2  # each request takes two seconds to answer
            return super().post(url, **kwargs)

    limited = FakeResponse(429, headers={"Retry-After": "5"})
    reply = _chat(TwoSecondEndpoint([limited, _reply("the answer")])).complete([])
    assert (reply.attempts, reply.request_seconds, reply.seconds) == (2, 2, 9)


def test_a_slowed_model_takes_its_extra_seconds_on_every_call_and_says_so_in_its_recipe():
    slept: list[float] = []
    slowed = SlowedChat(ScriptedChat(["one", "two"]), extra_seconds=3.0, sleep=slept.append)
    replies = [slowed.complete([]), slowed.complete([])]
    assert slept == [3.0, 3.0]
    assert replies == [Reply("one", seconds=3.0, request_seconds=3.0), Reply("two", seconds=3.0, request_seconds=3.0)]
    assert slowed.recipe() == {"model": "scripted", "extra_seconds": 3.0}


def test_a_refused_request_raises_with_the_endpoints_message():
    endpoint = FakeEndpoint([FakeResponse(404, text="no such model")])
    with pytest.raises(ChatError, match="404: no such model"):
        _chat(endpoint).complete([])
    assert len(endpoint.requests) == 1


def test_the_client_gives_up_after_its_retries():
    endpoint = FakeEndpoint([_rate_limited(), _rate_limited()])
    with pytest.raises(ChatError, match="after 2 tries"):
        _chat(endpoint, retries=2).complete([])


def test_the_recipe_and_the_repr_leave_out_the_api_key():
    chat = _chat(FakeEndpoint([]))
    assert "secret" not in repr(chat) and "secret" not in str(chat.recipe())


def test_llm_query_sends_the_images_then_the_prompt_to_the_named_model():
    check, ground = ScriptedChat(["from check"]), ScriptedChat(["from ground"])
    llm_query = llm_query_tool({"check": check, "ground": ground}, default="check", image_settings=ImageSettings())
    assert llm_query("where is the cup?", images=[np.zeros((4, 4, 3), np.uint8)], model="ground") == "from ground"
    (message,) = ground.prompts[0]
    assert [part["type"] for part in message["content"]] == ["image_url", "text"]
    assert message["content"][0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert llm_query("is it done?") == "from check"


def test_llm_query_names_the_models_it_knows_when_asked_for_another():
    llm_query = llm_query_tool({"check": ScriptedChat([])}, default="check", image_settings=ImageSettings())
    with pytest.raises(ValueError, match="use one of \\['check'\\]"):
        llm_query("q", model="planner")
