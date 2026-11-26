"""What a model's reply says to the harness: the cell to run, and why the endpoint stopped it.

The standard library only. The loop reads replies as they arrive and the trace viewer reads them
back from a run, by the same rules.
"""

import re
from collections.abc import Mapping
from enum import StrEnum
from typing import Any

CODE_BLOCK = re.compile(r"```(?:repl|python|py)?[ \t]*\n(.*?)```", re.DOTALL)


class FinishReason(StrEnum):
    """Why an OpenAI-compatible endpoint stopped writing a reply."""

    STOP = "stop"
    LENGTH = "length"
    """The reply hit the endpoint's token limit and was cut off."""
    CONTENT_FILTER = "content_filter"
    """The endpoint's content filter stopped the reply."""


def extract_code(reply: str) -> str | None:
    """The cell to run from a model's reply, or ``None`` when it holds none.

    TODO(johnny): decide what a reply with several code blocks means. This baseline runs the first
    and silently drops the rest. See README.md, "The one policy left open".
    """
    match = CODE_BLOCK.search(reply)
    return match.group(1) if match else None


def finish_reason(response: Mapping[str, Any] | None) -> str | None:
    """The finish reason of the first choice in an endpoint's ``response``, if it gave one."""
    choices = (response or {}).get("choices") or [{}]
    return choices[0].get("finish_reason")
