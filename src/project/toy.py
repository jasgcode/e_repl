"""A toy world for the harness: a shelf of colored boxes to photograph and pick from.

It has the shape of a robot task (a camera, one irreversible action, an error that tells the model
what would have been allowed) and needs no hardware, so the loop, the prompt, and the trajectory
format can be exercised offline. Replace it with your own tools.
"""

import random
from dataclasses import dataclass

import numpy as np

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

INSTRUCTIONS = f"""You operate a picking arm in front of a shelf through a persistent Python REPL.
The shelf holds a row of colored boxes in numbered slots, 0 at the left. You may pick exactly one
box, and a pick cannot be undone, so find the right slot before you pick. In shelf.photo() each
box is a {BOX_SIDE}x{BOX_SIDE} pixel square of one flat color."""


class PickRefused(Exception):
    """A pick the shelf did not carry out. The message is shown to the model."""


@dataclass
class Shelf:
    """A row of colored boxes."""

    colors: tuple[str, ...]
    picked: str | None = None
    """The color of the box that was taken."""

    def photo(self) -> np.ndarray:
        """A BGR uint8 image of the shelf: one flat-colored square per slot, left to right."""
        image = np.zeros((BOX_SIDE, BOX_SIDE * len(self.colors), 3), np.uint8)
        for slot, color in enumerate(self.colors):
            image[:, slot * BOX_SIDE : (slot + 1) * BOX_SIDE] = COLORS[color]
        return image

    def pick(self, slot: int) -> None:
        """Take the box in `slot` (0 is the leftmost). Only one pick is allowed."""
        if self.picked is not None:
            raise PickRefused("a box was already picked; only one pick is allowed, so call done()")
        if not 0 <= slot < len(self.colors):
            raise PickRefused(f"slot {slot} does not exist: use a slot from 0 to {len(self.colors) - 1}")
        self.picked = self.colors[slot]


def shuffled_shelf(seed: int) -> Shelf:
    """A shelf with one box of each color, in an order drawn from ``seed``."""
    colors = list(COLORS)
    random.Random(seed).shuffle(colors)
    return Shelf(tuple(colors))
