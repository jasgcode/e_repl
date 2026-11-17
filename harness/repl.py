"""A persistent Python namespace that a model drives one cell at a time.

The model has one action: run this code. Tools are ordinary Python objects in the namespace, so
the model composes them freely (loops, retries, numpy on arrays) and keeps bulky data in
variables. Only what the code prints (truncated), the value of a trailing expression, and images
passed to ``show()`` go back into the model's context.

This is not a sandbox. Model code runs in this process with this process's rights; a safety limit
belongs inside the tool that drives the hardware, where every call passes through it.
"""

import ast
import contextlib
import inspect
import io
import linecache
import signal
import threading
import time
import traceback
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import CodeType
from typing import Any

import numpy as np

from harness.images import ImageSettings, ObservedImage, encode_image

BUILTIN_NAMES = ("show", "done")
NO_OUTPUT = "(no output)"
NO_VARIABLES = "(no variables yet)"
SUMMARY_WIDTH = 80


class CellStatus(StrEnum):
    OK = "ok"
    ERROR = "error"


@dataclass(frozen=True)
class ReplConfig:
    """What the model sees of each cell. Part of a run's recipe."""

    max_output_chars: int = 4000
    max_images_per_cell: int = 4
    cell_timeout: float | None = 60.0
    """Seconds before a cell is interrupted; ``None`` lets cells run forever."""
    images: ImageSettings = field(default_factory=ImageSettings)


@dataclass(frozen=True)
class Observation:
    status: CellStatus
    text: str
    images: tuple[ObservedImage, ...]
    answer: str | None
    """What the cell passed to ``done()``; ``None`` if it did not call it."""

    @property
    def done(self) -> bool:
        return self.answer is not None


def truncate(text: str, limit: int) -> str:
    """Keep the head and the tail: errors and final results tend to sit at the end."""
    if len(text) <= limit:
        return text
    head = limit * 2 // 3
    cut = len(text) - limit
    note = f"\n... [{cut} chars truncated: keep big data in variables and print summaries] ...\n"
    return text[:head] + note + text[-(limit - head) :]


def summarize(value: Any, width: int = SUMMARY_WIDTH) -> str:
    """One line about a variable: its shape and type, not its contents."""
    if isinstance(value, np.ndarray):
        return f"ndarray shape={value.shape} dtype={value.dtype}"
    try:
        text = repr(value)
    except Exception as e:
        text = f"<{type(value).__name__}: repr failed: {e}>"
    if len(text) <= width:
        return text
    size = f" len={len(value)}" if hasattr(value, "__len__") else ""
    return f"{type(value).__name__}{size}: {text[: width - 30]}..."


def describe_tools(tools: Mapping[str, Any]) -> str:
    """The preloaded names as the system prompt lists them: signatures and first docstring lines.

    A tool's docstring is written for the model; its first line is all the model sees.
    """
    lines = []
    for name, obj in tools.items():
        if inspect.ismodule(obj):
            lines.append(f"- {name}: module {obj.__name__}")
        elif inspect.isroutine(obj):
            lines.append(f"- {name}{_signature(obj)}: {_first_line(obj)}")
        else:
            lines.append(f"- {name}: {type(obj).__name__}. {_first_line(obj)}")
            for attr, member in inspect.getmembers(type(obj), inspect.isfunction):
                if not attr.startswith("_"):
                    lines.append(f"    {name}.{attr}{_signature(member, drop_self=True)}: {_first_line(member)}")
    return "\n".join(lines)


def _signature(fn: Any, drop_self: bool = False) -> str:
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):  # builtins without introspectable signatures
        return "(...)"
    if drop_self:
        sig = sig.replace(parameters=list(sig.parameters.values())[1:])
    return str(sig)


def _first_line(obj: Any) -> str:
    doc = inspect.getdoc(obj)
    return doc.splitlines()[0] if doc else ""


@contextlib.contextmanager
def _time_limit(seconds: float | None) -> Iterator[None]:
    """Interrupt a runaway cell with SIGALRM, then hand any outer alarm its remaining time.

    SIGALRM only reaches the main thread; elsewhere cells run without a limit.
    """
    if seconds is None or threading.current_thread() is not threading.main_thread():
        yield
        return

    def on_alarm(signum: int, frame: Any) -> None:
        raise TimeoutError(f"cell exceeded the {seconds:g}s time limit")

    previous_handler = signal.signal(signal.SIGALRM, on_alarm)
    outer_remaining, outer_interval = signal.setitimer(signal.ITIMER_REAL, seconds)
    start = time.monotonic()
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if outer_remaining:
            left = max(outer_remaining - (time.monotonic() - start), 1e-3)
            signal.setitimer(signal.ITIMER_REAL, left, outer_interval)


class Repl:
    """A namespace that survives across cells, with ``show()`` and ``done()`` built in."""

    def __init__(self, tools: Mapping[str, Any], config: ReplConfig):
        clashes = set(tools) & set(BUILTIN_NAMES)
        if clashes:
            raise ValueError(f"tool names {sorted(clashes)} are reserved for the REPL's own builtins")
        self.tools = dict(tools)
        self.config = config
        self.namespace: dict[str, Any] = {"__name__": "__repl__", "show": self._show, "done": self._done}
        self.namespace.update(self.tools)
        self._hidden = set(self.namespace)
        self._cells = 0
        self._images: list[ObservedImage] = []
        self._answer: str | None = None

    def _show(self, image: np.ndarray, caption: str = "") -> None:
        """Attach an image (HxW gray or HxWx3 BGR, uint8) to this turn's observation."""
        if len(self._images) >= self.config.max_images_per_cell:
            print(f"[show: image dropped, at most {self.config.max_images_per_cell} per cell]")
            return
        self._images.append(encode_image(np.asarray(image), caption, self.config.images))

    def _done(self, answer: str = "") -> None:
        """Finish the task. Pass a short summary of what happened."""
        self._answer = str(answer)

    def variable_summary(self) -> str:
        """One line per variable the model defined, for the observation."""
        names = [
            name
            for name, value in self.namespace.items()
            if name not in self._hidden and not name.startswith("_") and not inspect.ismodule(value)
        ]
        if not names:
            return NO_VARIABLES
        return "\n".join(f"{name} = {summarize(self.namespace[name])}" for name in names)

    def run(self, code: str) -> Observation:
        """Run one cell. Any failure becomes an ERROR observation; only Ctrl-C escapes."""
        self._cells += 1
        filename = f"<cell {self._cells}>"
        # Register the source so tracebacks can show the failing line.
        linecache.cache[filename] = (len(code), None, code.splitlines(True), filename)
        self._images = []
        self._answer = None
        out = io.StringIO()
        status = CellStatus.OK
        try:
            body, tail = _compile_cell(code, filename)
            with (
                contextlib.redirect_stdout(out),
                contextlib.redirect_stderr(out),
                _time_limit(self.config.cell_timeout),
            ):
                exec(body, self.namespace)
                if tail is not None:
                    value = eval(tail, self.namespace)
                    if value is not None:
                        print(repr(value))
        except KeyboardInterrupt:
            raise  # Ctrl-C stops the whole agent, never just one cell of it
        except SystemExit:
            out.write("[exit() ignored: call done() to finish]\n")
        except BaseException as e:
            status = CellStatus.ERROR
            out.write(_format_error(e))
        return Observation(
            status=status,
            text=truncate(out.getvalue().rstrip(), self.config.max_output_chars) or NO_OUTPUT,
            images=tuple(self._images),
            answer=self._answer,
        )


def _compile_cell(code: str, filename: str) -> tuple[CodeType, CodeType | None]:
    """Split off a trailing expression so its value is echoed, as at the Python prompt."""
    tree = ast.parse(code, filename, mode="exec")
    last = tree.body[-1] if tree.body else None
    if not isinstance(last, ast.Expr):
        return compile(tree, filename, "exec"), None
    tree.body.pop()
    return compile(tree, filename, "exec"), compile(ast.Expression(last.value), filename, "eval")


def _format_error(e: BaseException) -> str:
    """A traceback without this module's frames, so the model sees its own code and the tools'."""
    if isinstance(e, SyntaxError):
        return "".join(traceback.format_exception_only(e))
    frames = [frame for frame in traceback.extract_tb(e.__traceback__) if frame.filename != __file__]
    text = "Traceback (most recent call last):\n" if frames else ""
    return text + "".join(traceback.format_list(frames)) + "".join(traceback.format_exception_only(e))
