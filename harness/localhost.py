"""A server on localhost, run from a background thread: what the viewer and every service share.

The standard library only.
"""

import contextlib
import logging
import threading
from collections.abc import Iterator, Mapping
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

logger = logging.getLogger(__name__)

HOST = "127.0.0.1"
ANY_FREE_PORT = 0
SHUTDOWN_POLL = 0.02
"""Seconds between the serving thread's checks for a shutdown; the standard library's 0.5 makes every stop wait."""


class QuietHandler(BaseHTTPRequestHandler):
    """A request handler that logs requests at debug level and sends a whole body in one call."""

    def send_body(
        self, status: HTTPStatus, content_type: str, body: bytes, headers: Mapping[str, str] | None = None
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 (the base class's name)
        logger.debug(format, *args)  # the base class writes every request to stderr


def server_url(server: HTTPServer) -> str:
    return f"http://{HOST}:{server.server_port}"


@contextlib.contextmanager
def serving(handler: type[BaseHTTPRequestHandler], port: int) -> Iterator[str]:
    """Serve ``handler`` on localhost from a background thread, one request at a time. Yields its URL."""
    server = HTTPServer((HOST, port), handler)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": SHUTDOWN_POLL}, daemon=True)
    thread.start()
    try:
        yield server_url(server)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
