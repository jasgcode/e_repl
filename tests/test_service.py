"""A device behind its service: what a client can ask for, when the device may move, and what the log keeps."""

import json
from collections.abc import Iterator

import pytest
import requests

from harness.localhost import ANY_FREE_PORT
from harness.service import ActionLog, Device, Service, ServiceCallFailed, ServiceClient, serving


class TooHigh(Exception):
    """The winch's own refusal."""


class Winch:
    """A stand-in for hardware: it lifts a load, refuses to go too high, and can jam."""

    def __init__(self) -> None:
        self.height = 0.0
        self.jammed = False

    def position(self) -> dict:
        return {"height": self.height}

    def lift(self, to: float) -> None:
        if to > 2.0:
            raise TooHigh("the winch lifts to 2.0 m at most")
        if self.jammed:
            raise RuntimeError("the cable is jammed")
        self.height = to

    def cut_power(self) -> None:
        self.height = 0.0  # the operator's: the load drops


class Clock:
    now = 0.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def winch() -> Winch:
    return Winch()


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def service(winch, clock, tmp_path) -> Service:
    listed, reads = ("position", "lift"), ("position",)
    device = Device(winch, listed, reads, refusals=(TooHigh, TypeError), reading=lambda: winch.position())
    return Service(device, ActionLog(tmp_path / "actions" / "session.jsonl"), clock)


@pytest.fixture
def url(service) -> Iterator[str]:
    with serving(service, ANY_FREE_PORT) as url:
        yield url


@pytest.fixture
def client(url) -> ServiceClient:
    return ServiceClient(url, raises={"TooHigh": TooHigh}, call_timeout=5.0)


def _lines(service: Service) -> list[dict]:
    return [json.loads(line) for line in service.log.path.read_text().splitlines()]


def test_a_listed_call_inside_a_run_moves_the_device_and_is_logged_with_its_readings(winch, service, client):
    with client.run(seconds=None):
        client.call("lift", to=1.5)
        assert client.call("position") == {"height": 1.5}
    opened, lifted, read, closed = _lines(service)
    assert [line["sequence"] for line in (opened, lifted, read, closed)] == [0, 1, 2, 3]
    assert (opened["event"], opened["seconds"], closed["event"]) == ("run_opened", None, "run_closed")
    assert (lifted["method"], lifted["arguments"], lifted["outcome"]) == ("lift", {"to": 1.5}, "ok")
    assert (lifted["before"], lifted["after"]) == ({"height": 0.0}, {"height": 1.5})
    assert read["outcome"] == "ok" and read["before"] is None  # a reading moves nothing
    assert winch.height == 1.5 and isinstance(client.last_service_seconds, float)


def test_the_device_does_not_move_outside_a_run_and_still_answers_readings(winch, service, client):
    with pytest.raises(ServiceCallFailed, match="RunClosed: no run is open"):
        client.call("lift", to=1.0)
    assert client.call("position") == {"height": 0.0} and winch.height == 0.0
    assert _lines(service)[0]["outcome"] == "outside_run"


def test_the_device_stops_moving_when_the_runs_time_limit_passes(winch, service, clock, client):
    client.open_run(seconds=30)
    client.call("lift", to=1.0)
    clock.now = 31
    with pytest.raises(ServiceCallFailed, match="the run's time limit has passed"):
        client.call("lift", to=1.5)
    assert winch.height == 1.0
    assert [line.get("outcome") for line in _lines(service)] == [None, "ok", "outside_run"]


def test_a_closed_run_leaves_the_device_still_and_two_runs_cannot_be_open_at_once(winch, client):
    with client.run(seconds=None):
        with pytest.raises(ServiceCallFailed, match="a run is already open"):
            client.open_run(seconds=None)
        client.call("lift", to=1.0)
    with pytest.raises(ServiceCallFailed, match="RunClosed"):
        client.call("lift", to=2.0)
    assert winch.height == 1.0


def test_what_the_device_refuses_comes_back_as_its_own_exception_and_stops_nothing(winch, service, client):
    with client.run(seconds=None):
        with pytest.raises(TooHigh, match=r"2\.0 m at most"):
            client.call("lift", to=3.0)
        client.call("lift", to=1.0)
    refused = _lines(service)[1]
    assert refused["outcome"] == "refused" and refused["raised"]["type"] == "TooHigh"
    assert refused["before"] == refused["after"] and service.latch.reason is None and winch.height == 1.0


def test_a_fault_stops_the_device_until_a_person_clears_it(winch, service, client):
    with client.run(seconds=None):
        winch.jammed = True
        with pytest.raises(ServiceCallFailed, match="RuntimeError: the cable is jammed"):
            client.call("lift", to=1.0)
        winch.jammed = False  # whatever jammed it is gone; nobody has looked yet
        with pytest.raises(ServiceCallFailed, match=r"DeviceFaulted: .*jammed.*A person must check it"):
            client.call("lift", to=1.0)
        assert client.call("position") == {"height": 0.0}
        service.clear_on_enter(["\n"])  # the operator presses Enter at the server
        client.call("lift", to=1.0)
    events = [line.get("outcome") or line["event"] for line in _lines(service)]
    assert events == ["run_opened", "fault", "blocked", "ok", "fault_cleared", "ok", "run_closed"]
    assert winch.height == 1.0


@pytest.mark.parametrize("method", ["cut_power", "height", "__class__", "_private"])
def test_a_call_the_service_does_not_list_is_refused_and_logged_as_it_was_asked(winch, service, url, method):
    winch.height = 1.0
    refused = requests.post(f"{url}/call", json={"method": method, "arguments": {}}, timeout=5)
    assert refused.status_code == 400 and winch.height == 1.0
    (line,) = _lines(service)
    assert line["outcome"] == "not_served" and f'"{method}"' in line["request"]


def test_a_request_that_is_not_json_is_refused_and_kept(service, url):
    assert requests.post(f"{url}/call", data=b"not json at all", timeout=5).status_code == 400
    assert _lines(service)[0]["request"] == "/call not json at all"


def test_a_device_that_cannot_be_read_is_logged_as_a_fault_with_no_readings(winch, service, client):
    def dead() -> dict:
        raise OSError("the port vanished")

    winch.position = dead  # the log's reading fails
    winch.jammed = True
    with client.run(seconds=None), pytest.raises(ServiceCallFailed, match="RuntimeError"):
        client.call("lift", to=1.0)
    fault = _lines(service)[1]
    assert fault["outcome"] == "fault" and fault["before"] is None and fault["after"] is None


def test_a_session_opens_with_the_calls_served_and_the_code_that_serves_them(service, tmp_path):
    service.log_start(repo=tmp_path, port="/dev/ttyFAKE")
    (start,) = _lines(service)
    assert (start["event"], start["methods"], start["port"]) == ("start", ["position", "lift"], "/dev/ttyFAKE")
    assert start["code"]["commit"] is None  # tmp_path is not a repository
