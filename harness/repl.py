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
import functools
import inspect
import io
import linecache
import signal
import threading
import time
import traceback
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from types import CodeType
from typing import Any, ClassVar

import numpy as np

from harness.images import ImageSettings, ObservedImage, encode_image
from harness.outcomes import CellStatus, Ending

BUILTIN_NAMES = ("show", "done", "give_up", "refuse")
NO_OUTPUT = "(no output)"
NO_VARIABLES = "(no variables yet)"
SUMMARY_WIDTH = 80
TIMED = "__repl_timed__"
"""Marks a method a REPL has already wrapped, so a tool held by several REPLs is wrapped once."""


@dataclass(frozen=True)
class ToolCall:
    """One call a cell made to a tool, as the REPL timed it."""

    name: str
    """The name the model used: ``robot.goto``, or ``llm_query`` for a function."""
    at: float
    """When the call began, in seconds since the REPL was built: the run's clock, near enough.

    With it, the age of what the model last saw when it acted can be worked out afterwards.
    """
    seconds: float
    raised: str | None
    """The type of what the call raised, or ``None`` when it returned."""
    facts: dict[str, Any] | None
    """What the tool itself said about this call: how old the frame was, how long the hardware took.

    A tool object reports them by setting its ``call_facts`` attribute to a dict during the call.
    """


@dataclass(frozen=True)
class ReplConfig:
    """What the model sees of each cell. Part of a run's recipe."""

    max_output_chars: int = 4000
    max_images_per_cell: int = 4
    cell_timeout: float | None = 60.0
    """Seconds before a cell is interrupted; ``None`` lets cells run forever.

    A backstop for runaway model code. It raises at whatever line is running, inside a tool or not,
    so a tool with long calls bounds them itself and this stays longer than the longest of them.
    """
    images: ImageSettings = field(default_factory=ImageSettings)


@dataclass(frozen=True)
class Observation:
    status: CellStatus
    text: str
    images: tuple[ObservedImage, ...]
    ending: Ending | None
    """Set when the cell called ``done()``, ``give_up()`` or ``refuse()``."""
    answer: str | None
    """What the cell passed to that call; ``None`` if it made none of them."""
    calls: tuple[ToolCall, ...]
    """The tool calls the cell made, in the order they ended. A tool's calls to itself are not listed."""


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


def public_methods(tool: Any) -> list[tuple[str, Callable[..., Any]]]:
    """The calls a tool object offers the model: the functions of its class not named with a leading ``_``."""
    return [(attr, fn) for attr, fn in inspect.getmembers(type(tool), inspect.isfunction) if not attr.startswith("_")]


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
            for attr, member in public_methods(obj):
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
def _time_limit(seconds: float | None, exceeded: str) -> Iterator[None]:
    """Interrupt a runaway cell with SIGALRM, then hand any outer alarm its remaining time.

    SIGALRM only reaches the main thread; elsewhere cells run without a limit.
    """
    if seconds is None or threading.current_thread() is not threading.main_thread():
        yield
        return

    def on_alarm(signum: int, frame: Any) -> None:
        raise TimeoutError(exceeded)

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


def _timed_method(owner: Any, attr: str, call: Callable[..., Any]) -> Callable[..., Any]:
    """``owner.attr`` timed for whichever REPL is running a cell. It holds no REPL, so it keeps none alive."""

    @functools.wraps(call)
    def timed(*args: Any, **kwargs: Any) -> Any:
        repl = Repl.active
        name = None if repl is None else repl.tool_names.get(id(owner))
        if repl is None or name is None:  # a call from outside any cell, or from a REPL that does not hold it
            return call(*args, **kwargs)
        return repl.timed_call(f"{name}.{attr}", owner, call, args, kwargs)

    setattr(timed, TIMED, True)
    return timed


class Repl:
    """A namespace that survives across cells, with ``show()`` and the three calls that end a run built in."""

    active: ClassVar["Repl | None"] = None
    """The REPL whose cell is running in this process. A tool call is timed for it alone."""

    def __init__(self, tools: Mapping[str, Any], config: ReplConfig):
        clashes = set(tools) & set(BUILTIN_NAMES)
        if clashes:
            raise ValueError(f"tool names {sorted(clashes)} are reserved for the REPL's own builtins")
        self.tools = dict(tools)
        self.config = config
        self.namespace: dict[str, Any] = {
            "__name__": "__repl__",
            "show": self._show,
            "done": self._done,
            "give_up": self._give_up,
            "refuse": self._refuse,
        }
        self.tool_names: dict[int, str] = {}
        """The name each tool object goes by in this namespace, by ``id``: how a timed call is labelled."""
        self._depth = 0
        self._born = time.monotonic()
        self.namespace.update({name: self._instrumented(name, tool) for name, tool in self.tools.items()})
        self._hidden = set(self.namespace)
        self._cells = 0
        self._new_cell()

    def _new_cell(self) -> None:
        """Forget what the last cell showed, called, and asked for."""
        self._images: list[ObservedImage] = []
        self._calls: list[ToolCall] = []
        self._ending: tuple[Ending, str] | None = None

    def _instrumented(self, name: str, tool: Any) -> Any:
        """``tool`` with its calls timed. An object stays the same object, so the model sees no wrapper."""
        if inspect.ismodule(tool):
            return tool
        if inspect.isroutine(tool):
            return self._timed_function(name, tool)
        self.tool_names[id(tool)] = name
        for attr, _ in public_methods(tool):
            if getattr(inspect.getattr_static(tool, attr), TIMED, False):
                continue  # another REPL holds this object too; a call is timed for the REPL whose cell made it
            try:
                setattr(tool, attr, _timed_method(tool, attr, getattr(tool, attr)))
            except AttributeError as e:  # slots, or a frozen dataclass
                raise TypeError(f"tool {name!r} cannot be timed: its methods cannot be replaced on the object") from e
        return tool

    def _timed_function(self, name: str, call: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(call)
        def timed(*args: Any, **kwargs: Any) -> Any:
            if Repl.active is not self:
                return call(*args, **kwargs)
            return self.timed_call(name, None, call, args, kwargs)

        return timed

    def timed_call(self, label: str, owner: Any, call: Callable[..., Any], args: tuple, kwargs: dict) -> Any:
        """Make one tool call from this REPL's running cell and list it with the cell's calls."""
        if self._depth:  # a tool calling its own methods is part of the call already being timed
            return call(*args, **kwargs)
        self._depth += 1
        start, raised = time.monotonic(), None
        try:
            return call(*args, **kwargs)
        except BaseException as e:
            raised = type(e).__name__
            raise
        finally:
            self._depth -= 1
            facts = getattr(owner, "call_facts", None)
            if isinstance(facts, dict):
                owner.call_facts = None  # each call reports its own
            else:
                facts = None
            self._calls.append(ToolCall(label, start - self._born, time.monotonic() - start, raised, facts))

    def _show(self, image: np.ndarray, caption: str = "") -> None:
        """Attach an image (HxW gray or HxWx3 BGR, uint8) to this turn's observation."""
        if len(self._images) >= self.config.max_images_per_cell:
            print(f"[show: image dropped, at most {self.config.max_images_per_cell} per cell]")
            return
        self._images.append(encode_image(np.asarray(image), caption, self.config.images))

    def _done(self, answer: str = "") -> None:
        """End the run: the task is complete. Pass a short summary of what happened."""
        self._ending = Ending.DONE, str(answer)

    def _give_up(self, reason: str) -> None:
        """End the run: the task cannot be done. Pass the reason."""
        self._ending = (
            Ending.GAVE_UP,
            _reason(reason, "give_up() needs a reason: say what stopped you, as in give_up('the bin is out of reach')"),
        )

    def _refuse(self, reason: str) -> None:
        """End the run: the task should not be done. Pass the reason."""
        self._ending = Ending.REFUSED, _reason(reason, "refuse() needs a reason: say why the task should not be done")

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

    def run(self, code: str, time_left: float | None = None) -> Observation:
        """Run one cell. Any failure becomes an ERROR observation; only Ctrl-C escapes.

        ``time_left`` is what remains of the run's own time limit, if it has one. The cell is
        stopped when that runs out, as it is at ``cell_timeout``.
        """
        limit = self.config.cell_timeout
        exceeded = "" if limit is None else f"cell exceeded the {limit:g}s time limit"
        if time_left is not None and (limit is None or time_left < limit):
            limit, exceeded = max(time_left, 1e-3), "the run's time limit ended this cell"
        self._cells += 1
        filename = f"<cell {self._cells}>"
        # Register the source so tracebacks can show the failing line.
        linecache.cache[filename] = (len(code), None, code.splitlines(True), filename)
        self._new_cell()
        out = io.StringIO()
        status = CellStatus.OK
        outer, Repl.active = Repl.active, self  # a cell can run another REPL's cell, as a sub-model's tool does
        try:
            body, tail = _compile_cell(code, filename)
            with (
                contextlib.redirect_stdout(out),
                contextlib.redirect_stderr(out),
                _time_limit(limit, exceeded),
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
        finally:
            Repl.active = outer
        ending, answer = self._ending or (None, None)
        return Observation(
            status=status,
            text=truncate(out.getvalue().rstrip(), self.config.max_output_chars) or NO_OUTPUT,
            images=tuple(self._images),
            ending=ending,
            answer=answer,
            calls=tuple(self._calls),
        )


def _reason(reason: str, missing: str) -> str:
    """``reason`` as text, or a ``ValueError`` saying ``missing`` when it is blank."""
    if not str(reason).strip():
        raise ValueError(missing)
    return str(reason)


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
