"""A toy world for the harness: a shelf of colored boxes to photograph and pick from.

It has the shape of a robot task (a camera, one irreversible action, an error that tells the model
what would have been allowed) and needs no hardware, so the loop, the prompt, and the trajectory
format can be exercised offline. Replace it with your own tools.

``Shelf`` is the world in the harness's process. ``project.shelf_service`` serves the same shelf
from a process of its own, and ``project.shelf_client`` is what the model then holds.
"""

import random
from dataclasses import dataclass, field

import numpy as np

from harness.outcomes import Verdict

BOX_SIDE = 40
"""Pixels per box in ``Shelf.photo``."""

# BGR, the order the harness expects from ``show()``.
COLORS = {
    "red": (0, 0, 255),
    "green": (0, 200, 0),
    "blue": (255, 0, 0),
    "yellow": (0, 220, 220),
    "white": (255, 255, 255),
}

# The world only: nothing here names the harness, so any harness can be given this text.
INSTRUCTIONS = f"""You operate a picking arm in front of a shelf. The shelf holds a row of colored
boxes in numbered slots, 0 at the left. You may pick exactly one box, and a pick cannot be undone,
so find the right slot before you pick. In shelf.photo() each box is a {BOX_SIDE}x{BOX_SIDE} pixel
square of one flat color."""


class PickRefused(Exception):
    """A pick the shelf did not carry out. The message is shown to the model."""


@dataclass
class Shelf:
    """A row of colored boxes."""

    colors: tuple[str, ...]
    _picked: str | None = field(default=None, init=False, repr=False)

    @property
    def picked(self) -> str | None:
        """The color of the box that was taken. The grader reads it, and only ``pick`` sets it.

        The REPL is not a sandbox: this stops a cell that assigns the attribute, not one that
        reaches for ``_picked``. Only a world in its own process keeps its state from the model.
        """
        return self._picked

    @picked.setter
    def picked(self, color: str) -> None:
        raise PickRefused("shelf.picked is read-only: a box is taken only by shelf.pick(slot)")

    def photo(self) -> np.ndarray:
        """A BGR uint8 image of the shelf: one flat-colored square per slot, left to right."""
        image = np.zeros((BOX_SIDE, BOX_SIDE * len(self.colors), 3), np.uint8)
        for slot, color in enumerate(self.colors):
            image[:, slot * BOX_SIDE : (slot + 1) * BOX_SIDE] = COLORS[color]
        return image

    def pick(self, slot: int) -> None:
        """Take the box in `slot` (0 is the leftmost). Only one pick is allowed."""
        if self._picked is not None:
            raise PickRefused("a box was already picked; only one pick is allowed, so call done()")
        if not 0 <= slot < len(self.colors):
            raise PickRefused(f"slot {slot} does not exist: use a slot from 0 to {len(self.colors) - 1}")
        self._picked = self.colors[slot]


def pick_verdict(shelf: Shelf, wanted: str) -> Verdict:
    """Whether the box taken from ``shelf`` is the ``wanted`` color.

    The experiment's grader. It is a function here and not a method of ``Shelf``, because the
    model is shown every public method of a tool and could call it.
    """
    if shelf.picked is None:
        return Verdict(False, f"no box was picked; the {wanted} box was wanted")
    if shelf.picked != wanted:
        return Verdict(False, f"the {shelf.picked} box was picked; the {wanted} box was wanted")
    return Verdict(True, f"the {wanted} box was picked")


def shuffled_shelf(seed: int) -> Shelf:
    """A shelf with one box of each color, in an order drawn from ``seed``."""
    colors = list(COLORS)
    random.Random(seed).shuffle(colors)
    return Shelf(tuple(colors))
