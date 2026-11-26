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

from harness.provenance import provenance
from harness.trajectory import utc_timestamp

logger = logging.getLogger(__name__)

HOST = "127.0.0.1"
ANY_FREE_PORT = 0
SHUTDOWN_POLL = 0.02
"""Seconds between the serving thread's checks for a shutdown; the standard library's 0.5 makes every stop wait."""
CALL_PATH = "/call"
RUN_PATH = "/run"
RAISED_STATUS = HTTPStatus.CONFLICT
"""The status of a call that was not carried out; the body is ``{RAISED_KEY: {"type", "message"}}``."""
RAISED_KEY = "raised"
RESULT_KEY = "result"
SECONDS_KEY = "seconds"
FAULTED = "DeviceFaulted"
RUN_CLOSED = "RunClosed"
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

    def call(self, started: float, method: str, arguments: dict[str, Any]) -> tuple[HTTPStatus, dict[str, Any]]:
        """Carry out one listed call, or decline it, and log which. Returns the answer to send."""
        clock = time.monotonic()
        moves = method not in self.device.reads

        def record(outcome: CallOutcome, **fields: Any) -> float:
            seconds = round(time.monotonic() - clock, 3)
            call = {"method": method, "arguments": arguments, "outcome": outcome}
            self.log.write(Event.CALL, started, **call, seconds=seconds, **fields)
            return seconds

        def declined(kind: str, message: str) -> tuple[HTTPStatus, dict[str, Any]]:
            return RAISED_STATUS, {RAISED_KEY: {"type": kind, "message": message}}

        fault = self.latch.reason
        if fault is not None and moves:
            record(CallOutcome.BLOCKED, fault=fault)
            return declined(
                FAULTED,
                f"the device is stopped after a fault ({fault}). A person must check it and clear the fault at "
                "its server before it moves again. Readings still answer.",
            )
        closed = self.run.why_closed()
        if closed is not None and moves:
            record(CallOutcome.OUTSIDE_RUN, why=closed)
            return declined(RUN_CLOSED, f"{closed}. Readings still answer.")
        before = self._reading() if moves else None
        try:
            result = getattr(self.device.target, method)(**arguments)
        except Exception as e:
            # What the device raised is the call's answer, and the client raises it again on its side.
            # A refusal leaves the device free; anything else stops it until a person has looked.
            raised = {"type": type(e).__name__, "message": str(e)}
            after = self._reading() if moves else None
            if isinstance(e, self.device.refusals):
                logger.warning("%s(%s) refused: %s: %s", method, arguments, raised["type"], e)
                record(CallOutcome.REFUSED, raised=raised, before=before, after=after)
            else:
                self.latch.trip(f"{method} raised {raised['type']}: {e}")
                logger.error("DEVICE FAULT: %s(%s) raised %s: %s", method, arguments, raised["type"], e)
                logger.error("Every motion is refused. Check the device, then press Enter here to let it move.")
                record(CallOutcome.FAULT, raised=raised, before=before, after=after)
            return RAISED_STATUS, {RAISED_KEY: raised}
        seconds = record(CallOutcome.OK, result=result, before=before, after=self._reading() if moves else None)
        return HTTPStatus.OK, {RESULT_KEY: result, SECONDS_KEY: seconds}


def _handler(service: Service) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            started = time.time()
            raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            try:
                body = json.loads(raw or b"null")
            except json.JSONDecodeError:
                body = None
            if self.path == RUN_PATH:
                self._open_run(started, body)
                return
            method = body.get("method") if isinstance(body, dict) else None
            arguments = body.get("arguments", {}) if isinstance(body, dict) else None
            listed = service.device.methods
            if self.path != CALL_PATH or method not in listed or not isinstance(arguments, dict):
                request = f"{self.path} {raw[:LOGGED_REQUEST_CHARS].decode(errors='replace')}"
                service.log.write(Event.CALL, started, outcome=CallOutcome.NOT_SERVED, request=request)
                logger.warning("not served: %s", request)
                self.send_error(HTTPStatus.BAD_REQUEST, f"a call names one of {', '.join(listed)}")
                return
            self._reply(*service.call(started, method, arguments))

        def do_DELETE(self) -> None:
            if self.path != RUN_PATH:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            service.run.close()
            service.log.write(Event.RUN_CLOSED, time.time())
            self._reply(HTTPStatus.OK, {})

        def _open_run(self, started: float, body: Any) -> None:
            seconds = body.get("seconds") if isinstance(body, dict) else None
            if seconds is not None and (
                isinstance(seconds, bool) or not isinstance(seconds, int | float) or seconds <= 0
            ):
                self.send_error(HTTPStatus.BAD_REQUEST, "a run's seconds are a number above zero, or null for no limit")
                return
            try:
                service.run.open(seconds)
            except RuntimeError as e:
                self._reply(RAISED_STATUS, {RAISED_KEY: {"type": RUN_CLOSED, "message": str(e)}})
                return
            service.log.write(Event.RUN_OPENED, started, seconds=seconds)
            self._reply(HTTPStatus.OK, {})

        def _reply(self, status: HTTPStatus, answer: dict[str, Any]) -> None:
            body = json.dumps(answer).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 (the base class's name)
            logger.debug(format, *args)  # the base class writes every request to stderr

    return Handler


def server(service: Service, port: int) -> HTTPServer:
    """A server for ``service`` bound to localhost. One thread, so one call at a time."""
    return HTTPServer((HOST, port), _handler(service))


def server_url(http_server: HTTPServer) -> str:
    return f"http://{HOST}:{http_server.server_port}"


@contextlib.contextmanager
def serving(service: Service, port: int) -> Iterator[str]:
    """Serve ``service`` from a background thread. Yields its URL."""
    http_server = server(service, port)
    thread = threading.Thread(target=http_server.serve_forever, kwargs={"poll_interval": SHUTDOWN_POLL}, daemon=True)
    thread.start()
    try:
        yield server_url(http_server)
    finally:
        http_server.shutdown()
        http_server.server_close()
        thread.join()


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
        self.last_service_seconds: float | None = None
        """How long the service took over the last call it carried out, by its own clock."""

    def call(self, method: str, **arguments: Any) -> Any:
        """Ask the service for ``method``. Raises what the device raised, or ``ServiceCallFailed``."""
        self.last_service_seconds = None
        body = {"method": method, "arguments": arguments}
        try:
            response = requests.post(self._url + CALL_PATH, json=body, timeout=(CONNECT_TIMEOUT, self._call_timeout))
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
        response = requests.post(self._url + RUN_PATH, json={"seconds": seconds}, timeout=CONNECT_TIMEOUT)
        if response.status_code == RAISED_STATUS:
            raise ServiceCallFailed(response.json()[RAISED_KEY]["message"])
        response.raise_for_status()

    def close_run(self) -> None:
        """End the run. From here the device refuses to move, so it is still when it is graded."""
        requests.delete(self._url + RUN_PATH, timeout=CONNECT_TIMEOUT).raise_for_status()

    @contextlib.contextmanager
    def run(self, seconds: float | None) -> Iterator[None]:
        """A run for the length of a ``with`` block, closed on the way out whatever happened."""
        self.open_run(seconds)
        try:
            yield
        finally:
            self.close_run()
