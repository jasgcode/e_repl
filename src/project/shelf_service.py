"""The toy shelf as a service: the world in its own process, reached over HTTP on localhost.

    uv run python -m project.shelf_service --seed 0 --wanted red

The service has two channels, for two parties.

- An agent's code holds a ``ShelfClient`` and reaches ``GET /photo`` and ``POST /pick``. Every other
  path is 404: none returns the colors, the wanted color, or the verdict.
- The process that started the service holds its pipes. The first line the service prints is its
  URL. Closing its stdin ends the episode: every later pick is refused, and the last line the
  service prints is the verdict.

So the model's code cannot write the shelf's state, and the verdict reaches only the owner. This
is still not a sandbox: a cell runs with the owner's rights and could attack the owner's process.
What the boundary removes is the state a cell could change by assigning an attribute.

A service that drives hardware is started on its own, so that it outlives a harness that crashes.
This one ends with its owner, so that a test or a stopped run leaves no process behind.
"""

import contextlib
import dataclasses
import io
import json
import logging
import subprocess
import sys
import threading
from collections.abc import Iterator
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from typing import IO, Any

import click
import numpy as np

from harness import localhost
from harness.localhost import ANY_FREE_PORT, QuietHandler
from harness.outcomes import Verdict
from project.shelf_client import PHOTO_PATH, PICK_PATH, REFUSED_KEY, REFUSED_STATUS
from project.toy import COLORS, PickRefused, Shelf, pick_verdict, shuffled_shelf

logger = logging.getLogger(__name__)

URL_KEY = "url"
EPISODE_OVER = "the episode is over, so nothing more can be picked; call done()"


class Episode:
    """One shelf and the color wanted from it. Every request and the verdict take its lock."""

    def __init__(self, shelf: Shelf, wanted: str):
        self._shelf = shelf
        self._wanted = wanted
        self._lock = threading.Lock()
        self._verdict: Verdict | None = None

    def photo(self) -> np.ndarray:
        with self._lock:
            return self._shelf.photo()

    def pick(self, slot: int) -> None:
        with self._lock:
            if self._verdict is not None:
                raise PickRefused(EPISODE_OVER)
            self._shelf.pick(slot)

    def finish(self) -> Verdict:
        """End the episode and grade it. Every later pick is refused, so the verdict stands."""
        with self._lock:
            if self._verdict is None:
                self._verdict = pick_verdict(self._shelf, self._wanted)
            return self._verdict


def _handler(episode: Episode) -> type[BaseHTTPRequestHandler]:
    class Handler(QuietHandler):
        def do_GET(self) -> None:
            if self.path != PHOTO_PATH:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            photo = io.BytesIO()
            np.save(photo, episode.photo())
            self.send_body(HTTPStatus.OK, "application/octet-stream", photo.getvalue())

        def do_POST(self) -> None:
            if self.path != PICK_PATH:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            try:
                episode.pick(_slot(body))
            except PickRefused as e:
                refusal = json.dumps({REFUSED_KEY: str(e)}).encode()
                self.send_body(REFUSED_STATUS, "application/json", refusal)
                return
            self.send_body(HTTPStatus.NO_CONTENT, "application/json", b"")

    return Handler


def _slot(body: Any) -> int:
    """The slot a pick request names. The wire can carry anything, so this is where it is checked."""
    slot = body.get("slot") if isinstance(body, dict) else None
    if not isinstance(slot, int) or isinstance(slot, bool):
        raise PickRefused(f'a pick names one slot as {{"slot": <integer>}}, got {body!r}')
    return slot


def serving(episode: Episode, port: int) -> contextlib.AbstractContextManager[str]:
    """Serve ``episode`` on localhost from a background thread. Yields the service's URL."""
    return localhost.serving(_handler(episode), port)


class ShelfService:
    """A running service, as the process that started it holds it."""

    def __init__(self, process: subprocess.Popen[str]):
        assert process.stdin is not None and process.stdout is not None
        self._process = process
        self._stdin: IO[str] = process.stdin
        self._stdout: IO[str] = process.stdout
        self._verdict: Verdict | None = None
        self.url: str = self._answer()[URL_KEY]

    def finish(self) -> Verdict:
        """End the episode and return the service's verdict. This is the ``grade`` of a run."""
        if self._verdict is None:
            self._stdin.close()
            self._verdict = Verdict(**self._answer())
        return self._verdict

    def _answer(self) -> dict[str, Any]:
        line = self._stdout.readline()
        if not line:
            raise RuntimeError(f"the shelf service exited with code {self._process.wait()} before it answered")
        return json.loads(line)


@contextlib.contextmanager
def running_shelf_service(seed: int, wanted: str) -> Iterator[ShelfService]:
    """Start the service as a child process and end it, graded or not, on the way out."""
    command = [sys.executable, "-m", "project.shelf_service", "--seed", str(seed), "--wanted", wanted]
    # A session of its own: Ctrl-C in the terminal reaches the harness alone, and the service ends
    # when its stdin closes, which also happens if the harness dies.
    with subprocess.Popen(
        command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, start_new_session=True
    ) as process:
        service = ShelfService(process)
        try:
            yield service
        finally:
            service.finish()


@click.command()
@click.option("--seed", type=int, required=True, help="Orders the boxes on the shelf.")
@click.option("--wanted", type=click.Choice(sorted(COLORS)), required=True, help="The color the task asks for.")
@click.option("--port", type=int, default=ANY_FREE_PORT, show_default=True, help="0 takes any free port.")
def main(seed: int, wanted: str, port: int) -> None:
    """Serve one shelf. Prints the URL, waits for stdin to close, then prints the verdict."""
    episode = Episode(shuffled_shelf(seed), wanted)
    with serving(episode, port) as url:
        click.echo(json.dumps({URL_KEY: url}))
        sys.stdin.read()
        click.echo(json.dumps(dataclasses.asdict(episode.finish())))


if __name__ == "__main__":
    main()
