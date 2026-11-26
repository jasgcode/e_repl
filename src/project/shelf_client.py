"""The agent's side of the shelf service: the calls of ``project.toy.Shelf``, sent over HTTP.

A client holds a URL and no shelf, so code that holds one reaches the world only through the two
paths below. Any harness can drive the same world by importing this module.
"""

import io
import operator
from http import HTTPStatus

import numpy as np
import requests

from project.toy import PickRefused

PHOTO_PATH = "/photo"
PICK_PATH = "/pick"
REFUSED_STATUS = HTTPStatus.CONFLICT
"""The status of a pick the shelf did not carry out; the body is ``{REFUSED_KEY: message}``."""
REFUSED_KEY = "refused"
REQUEST_TIMEOUT = 10.0


class ShelfClient:
    """A row of colored boxes."""

    def __init__(self, url: str):
        self._url = url

    def photo(self) -> np.ndarray:
        """A BGR uint8 image of the shelf: one flat-colored square per slot, left to right."""
        response = requests.get(self._url + PHOTO_PATH, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        return np.load(io.BytesIO(response.content))

    def pick(self, slot: int) -> None:
        """Take the box in `slot` (0 is the leftmost). Only one pick is allowed."""
        # operator.index turns a numpy integer into an int, which JSON needs, and refuses a float.
        body = {"slot": operator.index(slot)}
        response = requests.post(self._url + PICK_PATH, json=body, timeout=REQUEST_TIMEOUT)
        if response.status_code == REFUSED_STATUS:
            raise PickRefused(response.json()[REFUSED_KEY])
        response.raise_for_status()
