"""Hardware as a service: one process holds the device, another asks it for the calls it lists.

This is for real hardware. The process that holds the device runs `server`; the agent's process
holds a `ServiceClient` and reaches `POST /call`, which runs one listed method on the device.
Whatever limit the device's own code enforces is enforced here, where the device is, and the
agent's code holds nothing it could use to reach around it.

Four things are true of every service, whatever the device:

- **Only listed calls run.** A request for anything else is refused and written down.
- **The device moves only inside a run.** A run is opened with its time limit and closed when it
  ends. Outside one, and after the limit has passed, every call that is not a reading is refused:
  the device is still when the run is graded, and nothing a model left running can move it.
- **A fault stops the device.** A call that raises anything but a listed refusal trips a latch,
  and nothing moves again until a person clears it at the server's terminal.
- **Everything is logged before it is answered**, by the process that holds the device: each call
  with its arguments, its outcome, its seconds, and the device's reading before and after.

The service handles one call at a time and answers when the call has ended. A client that stops
waiting does not stop the device.
"""

import contextlib
import dataclasses
import json
import logging
import os
import threading
import time
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from enum import StrEnum
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import requests

from harness import localhost
from harness.localhost import HOST, QuietHandler
from harness.provenance import provenance, utc_timestamp

logger = logging.getLogger(__name__)

CALL_PATH = "/call"
RUN_PATH = "/run"
RAISED_STATUS = HTTPStatus.CONFLICT
"""The status of a call that was not carried out; the body is ``{RAISED_KEY: {"type", "message"}}``."""
RAISED_KEY = "raised"
RESULT_KEY = "result"
SECONDS_KEY = "seconds"
FAULTED = "DeviceFaulted"
RUN_CLOSED = "RunClosed"
NOT_SERVED = "NotServed"
LOGGED_REQUEST_CHARS = 500
"""How much of a request the service could not read is kept in the log."""
CONNECT_TIMEOUT = 5.0


class Event(StrEnum):
    START = "start"
    """The server started: which calls it serves and which code it runs."""
    RUN_OPENED = "run_opened"
    RUN_CLOSED = "run_closed"
    CALL = "call"
    FAULT_CLEARED = "fault_cleared"
    """The operator pressed Enter at the server to let a faulted device move again."""


class CallOutcome(StrEnum):
    OK = "ok"
    REFUSED = "refused"
    """The device declined the call before anything moved."""
    FAULT = "fault"
    """The call raised anything else, and the device is stopped until a person clears it."""
    BLOCKED = "blocked"
    """Not attempted: the device was already stopped after a fault."""
    OUTSIDE_RUN = "outside_run"
    """Not attempted: no run was open, or the open run's time limit had passed."""
    NOT_SERVED = "not_served"
    """Not one of the calls the service lists, or not a request it could read."""


ANSWER_STATUS = {CallOutcome.OK: HTTPStatus.OK, CallOutcome.NOT_SERVED: HTTPStatus.BAD_REQUEST}
"""The HTTP status of each kind of answer. Every other outcome is ``RAISED_STATUS``."""


@dataclass(frozen=True)
class Answer:
    """What the service says to one request: how it went, and the body to send back."""

    outcome: CallOutcome
    body: dict[str, Any]
    """``{RESULT_KEY, SECONDS_KEY}`` for a call carried out, ``{RAISED_KEY: {"type", "message"}}`` otherwise."""


def _raised(kind: str, message: str) -> dict[str, Any]:
    return {RAISED_KEY: {"type": kind, "message": message}}


def _no_reading() -> None:
    return None


class ActionLog:
    """An append-only file of numbered lines. Each write reaches the disk before it returns."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._lock = threading.Lock()
        self._sequence = 0

    def write(self, event: Event, unix_time: float, **fields: Any) -> None:
        """Append one line: its number in this file, ``unix_time`` in UTC, the event, then ``fields``."""
        with self._lock, open(self.path, "a") as file:
            record = {"sequence": self._sequence, "time": utc_timestamp(unix_time), "event": event, **fields}
            file.write(json.dumps(record) + "\n")
            file.flush()
            os.fsync(file.fileno())
            self._sequence += 1


class FaultLatch:
    """Whether the device is stopped after a fault. The service trips it; only a person clears it."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._reason: str | None = None

    @property
    def reason(self) -> str | None:
        """What tripped the latch, or ``None`` while the device may move."""
        with self._lock:
            return self._reason

    def trip(self, reason: str) -> None:
        with self._lock:
            self._reason = self._reason or reason  # the first fault is the one to look at

    def clear(self) -> None:
        with self._lock:
            self._reason = None


class RunWindow:
    """The one run the device is serving, and when its time is up. The device moves only inside it."""

    def __init__(self, clock: Callable[[], float]):
        self._clock = clock
        self._lock = threading.Lock()
        self._open = False
        self._deadline: float | None = None

    def open(self, seconds: float | None) -> None:
        """Start a run that may move the device for ``seconds``, or with no limit when ``None``."""
        with self._lock:
            if self._open:
                raise RuntimeError("a run is already open: close it before another starts")
            self._open = True
            self._deadline = None if seconds is None else self._clock() + seconds

    def close(self) -> None:
        with self._lock:
            self._open, self._deadline = False, None

    def why_closed(self) -> str | None:
        """Why the device may not move now, or ``None`` when it may."""
        with self._lock:
            if not self._open:
                return "no run is open, so the device does not move"
            if self._deadline is not None and self._clock() >= self._deadline:
                return "the run's time limit has passed, so the device does not move"
            return None


@dataclass(frozen=True)
class Device:
    """What a service holds and how it treats a call to it."""

    target: Any
    """The object that drives the hardware. Its own limits apply to every call."""
    methods: tuple[str, ...]
    """The calls a client may make. Everything else on ``target`` is the operator's."""
    reads: tuple[str, ...]
    """The listed calls that move nothing. They are answered outside a run and while faulted."""
    refusals: tuple[type[BaseException], ...]
    """What ``target`` raises to decline a call before anything moves. Anything else it raises is a fault."""
    reading: Callable[[], dict[str, Any]]
    """The device's own state, for the log: read before and after every call that can move it."""


class Service:
    """One device being served: its log, its fault latch, and its run."""

    def __init__(self, device: Device, log: ActionLog, clock: Callable[[], float] = time.monotonic):
        self.device = device
        self.log = log
        self.latch = FaultLatch()
        self.run = RunWindow(clock)

    def log_start(self, repo: Path, **facts: Any) -> None:
        """Open the log's session: the calls served, the code in ``repo`` that serves them, and ``facts``."""
        code = dataclasses.asdict(provenance(repo))
        self.log.write(Event.START, time.time(), methods=list(self.device.methods), code=code, **facts)

    def clear_on_enter(self, lines: Iterable[str]) -> None:
        """Clear a fault each time a line arrives: the operator pressing Enter at the server."""
        for _ in lines:
            fault = self.latch.reason
            if fault is None:
                logger.info("no fault to clear")
                continue
            self.latch.clear()
            self.log.write(Event.FAULT_CLEARED, time.time(), fault=fault)
            logger.warning("fault cleared by the operator: %s", fault)

    def _reading(self) -> dict[str, Any] | None:
        try:
            return self.device.reading()
        except Exception as e:
            # The log describes a call and must never change its answer; the call itself reports a dead device.
            logger.warning("could not read the device for the log: %s: %s", type(e).__name__, e)
            return None

    def not_served(self, started: float, request: str) -> Answer:
        """Decline a request that names no listed call, and log it as it was asked."""
        self.log.write(Event.CALL, started, outcome=CallOutcome.NOT_SERVED, request=request)
        logger.warning("not served: %s", request)
        return Answer(
            CallOutcome.NOT_SERVED, _raised(NOT_SERVED, f"a call names one of {', '.join(self.device.methods)}")
        )

    def open_run(self, started: float, seconds: Any) -> Answer:
        """Let the device move for ``seconds``, or with no limit when ``None``, until the run is closed."""
        if seconds is not None and (isinstance(seconds, bool) or not isinstance(seconds, int | float) or seconds <= 0):
            return self.not_served(started, f"open a run for {seconds!r} seconds")
        try:
            self.run.open(seconds)
        except RuntimeError as e:
            return Answer(CallOutcome.REFUSED, _raised(RUN_CLOSED, str(e)))
        self.log.write(Event.RUN_OPENED, started, seconds=seconds)
        return Answer(CallOutcome.OK, {})

    def close_run(self) -> Answer:
        """End the run. From here the device refuses to move, so it is still when it is graded."""
        self.run.close()
        self.log.write(Event.RUN_CLOSED, time.time())
        return Answer(CallOutcome.OK, {})

    def call(self, started: float, method: str, arguments: Mapping[str, Any]) -> Answer:
        """Carry out one listed call, or decline it, and log which before answering."""
        if method not in self.device.methods:
            return self.not_served(started, json.dumps({"method": method, "arguments": arguments}))
        moves = method not in self.device.reads
        reading = self._reading if moves else _no_reading  # a reading moves nothing, so has no before and after

        def record(outcome: CallOutcome, seconds: float = 0.0, **fields: Any) -> None:
            call = {"method": method, "arguments": arguments, "outcome": outcome}
            self.log.write(Event.CALL, started, **call, seconds=round(seconds, 3), **fields)

        fault = self.latch.reason
        if fault is not None and moves:
            record(CallOutcome.BLOCKED, fault=fault)
            message = (
                f"the device is stopped after a fault ({fault}). A person must check it and clear the fault at "
                "its server before it moves again. Readings still answer."
            )
            return Answer(CallOutcome.BLOCKED, _raised(FAULTED, message))
        closed = self.run.why_closed()
        if closed is not None and moves:
            record(CallOutcome.OUTSIDE_RUN, why=closed)
            return Answer(CallOutcome.OUTSIDE_RUN, _raised(RUN_CLOSED, f"{closed}. Readings still answer."))
        before = reading()
        start = time.monotonic()  # the device's own time: the readings for the log are not part of it
        try:
            result = getattr(self.device.target, method)(**arguments)
        except Exception as e:
            # What the device raised is the call's answer, and the client raises it again on its side.
            # A refusal leaves the device free; anything else stops it until a person has looked.
            seconds = time.monotonic() - start
            raised = {"type": type(e).__name__, "message": str(e)}
            outcome = CallOutcome.REFUSED if isinstance(e, self.device.refusals) else CallOutcome.FAULT
            if outcome is CallOutcome.REFUSED:
                logger.warning("%s(%s) refused: %s: %s", method, arguments, raised["type"], e)
            else:
                self.latch.trip(f"{method} raised {raised['type']}: {e}")
                logger.error("DEVICE FAULT: %s(%s) raised %s: %s", method, arguments, raised["type"], e)
                logger.error("Every motion is refused. Check the device, then press Enter here to let it move.")
            record(outcome, seconds, raised=raised, before=before, after=reading())
            return Answer(outcome, {RAISED_KEY: raised})
        seconds = time.monotonic() - start
        record(CallOutcome.OK, seconds, result=result, before=before, after=reading())
        return Answer(CallOutcome.OK, {RESULT_KEY: result, SECONDS_KEY: round(seconds, 3)})


def _handler(service: Service) -> type[BaseHTTPRequestHandler]:
    """The wire: reads a request, hands it to ``service``, and sends its answer as JSON."""

    class Handler(QuietHandler):
        def do_POST(self) -> None:
            started = time.time()
            raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            try:
                body = json.loads(raw or b"null")
            except json.JSONDecodeError:
                body = None
            fields = body if isinstance(body, dict) else {}
            method, arguments = fields.get("method"), fields.get("arguments", {})
            if self.path == RUN_PATH:
                answer = service.open_run(started, fields.get("seconds"))
            elif self.path == CALL_PATH and isinstance(method, str) and isinstance(arguments, dict):
                answer = service.call(started, method, arguments)
            else:
                answer = service.not_served(
                    started, f"{self.path} {raw[:LOGGED_REQUEST_CHARS].decode(errors='replace')}"
                )
            self._reply(answer)

        def do_DELETE(self) -> None:
            if self.path != RUN_PATH:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            self._reply(service.close_run())

        def _reply(self, answer: Answer) -> None:
            status = ANSWER_STATUS.get(answer.outcome, RAISED_STATUS)
            self.send_body(status, "application/json", json.dumps(answer.body).encode())

    return Handler


def server(service: Service, port: int) -> HTTPServer:
    """A server for ``service`` bound to localhost. One thread, so one call at a time."""
    return HTTPServer((HOST, port), _handler(service))


def serving(service: Service, port: int) -> contextlib.AbstractContextManager[str]:
    """Serve ``service`` from a background thread. Yields its URL."""
    return localhost.serving(_handler(service), port)


class ServiceCallFailed(RuntimeError):
    """A call the service did not carry out. The message starts with what the service raised."""


class ServiceClient:
    """The other side of a service: asks for calls, and opens and closes the run.

    A tool the model holds wraps one of these and shows the model only the device's own calls.
    Opening and closing a run are the experiment's, never the model's.
    """

    def __init__(self, url: str, raises: Mapping[str, type[Exception]], call_timeout: float):
        """
        Args:
            url: Where the service listens.
            raises: Exception types to raise again here by the name the service reports, so a
                refusal reaches the model as the device's own exception.
            call_timeout: Seconds to wait for one call's answer. The service answers when the call has ended.
        """
        self._url = url
        self._raises = dict(raises)
        self._call_timeout = call_timeout
        self._session = requests.Session()
        self._session.trust_env = False  # localhost: no proxy to look up on every call
        self.last_service_seconds: float | None = None
        """How long the device took over the last call the service carried out, by the service's clock."""

    def call(self, method: str, **arguments: Any) -> Any:
        """Ask the service for ``method``. Raises what the device raised, or ``ServiceCallFailed``."""
        self.last_service_seconds = None
        body = {"method": method, "arguments": arguments}
        try:
            response = self._session.post(
                self._url + CALL_PATH, json=body, timeout=(CONNECT_TIMEOUT, self._call_timeout)
            )
        except requests.ReadTimeout as e:
            raise ServiceCallFailed(
                f"the service did not answer {method}() within {self._call_timeout:g} s; the call may still be "
                "under way, and a reading answers once it has ended"
            ) from e
        if response.status_code == RAISED_STATUS:
            raised = response.json()[RAISED_KEY]
            known = self._raises.get(raised["type"])
            if known is not None:
                raise known(raised["message"])
            raise ServiceCallFailed(f"{raised['type']}: {raised['message']}")
        response.raise_for_status()
        answer = response.json()
        self.last_service_seconds = answer[SECONDS_KEY]
        return answer[RESULT_KEY]

    def open_run(self, seconds: float | None) -> None:
        """Start a run: the device may move for ``seconds``, or without a limit when ``None``."""
        response = self._session.post(self._url + RUN_PATH, json={"seconds": seconds}, timeout=CONNECT_TIMEOUT)
        if response.status_code == RAISED_STATUS:
            raise ServiceCallFailed(response.json()[RAISED_KEY]["message"])
        response.raise_for_status()

    def close_run(self) -> None:
        """End the run. From here the device refuses to move, so it is still when it is graded."""
        self._session.delete(self._url + RUN_PATH, timeout=CONNECT_TIMEOUT).raise_for_status()

    @contextlib.contextmanager
    def run(self, seconds: float | None) -> Iterator[None]:
        """A run for the length of a ``with`` block, closed on the way out whatever happened."""
        self.open_run(seconds)
        try:
            yield
        finally:
            self.close_run()
