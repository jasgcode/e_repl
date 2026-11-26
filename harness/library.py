"""A library of named texts an agent keeps from one round of a campaign to the next.

The agent holds a ``Library`` as a tool and writes programs, skills, or notes into it. Whether a
round's writing is kept is the campaign's decision, so ``snapshot`` and ``restore`` are functions
outside the tool: every public method of a tool is a call the model may make.

The standard library only. The REPL is not a sandbox: a cell can reach ``_texts``, and a campaign
restores the library after a rejected round whatever the cell did to it.
"""

LibraryTexts = dict[str, str]


class LibraryError(Exception):
    """A library call that did nothing. The message is shown to the model."""


class Library:
    """Named texts that outlive a round: programs, skills, notes."""

    def __init__(self) -> None:
        self._texts: LibraryTexts = {}

    def write(self, name: str, text: str) -> None:
        """Store `text` under `name`, replacing it. Kept for later rounds only if this round is judged to work."""
        if not name.strip():
            raise LibraryError("a library entry needs a name that is not empty")
        if not isinstance(text, str):
            raise LibraryError(f"library entries are text; got {type(text).__name__} for {name!r}")
        self._texts[name] = text

    def read(self, name: str) -> str:
        """The text stored under `name`."""
        if name not in self._texts:
            raise LibraryError(f"no entry {name!r}; the library holds {sorted(self._texts)}")
        return self._texts[name]

    def names(self) -> list[str]:
        """The names of every entry, sorted."""
        return sorted(self._texts)


def snapshot(library: Library) -> LibraryTexts:
    """A copy of everything in ``library``."""
    return dict(library._texts)


def restore(library: Library, texts: LibraryTexts) -> None:
    """Put ``library`` back to ``texts``, dropping whatever was written since."""
    library._texts = dict(texts)
