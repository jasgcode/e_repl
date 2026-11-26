"""Where a run came from: the code's commit, whether the tree was clean, and the command line."""

import datetime
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Provenance:
    commit: str | None
    """``None`` outside a git repository."""
    uncommitted: tuple[str, ...]
    """``git status --porcelain`` lines; empty for a clean tree."""
    command: tuple[str, ...]
    started: str
    """ISO 8601, local time with its UTC offset."""
    machine: str
    """The operating system, processor type, and Python the run was timed on. A time measured on
    one machine is not the time on another."""


def utc_timestamp(unix_time: float) -> str:
    """``unix_time`` as an ISO 8601 UTC time to the millisecond, the form every record uses."""
    return datetime.datetime.fromtimestamp(unix_time, datetime.UTC).isoformat(timespec="milliseconds")


def _git(repo: Path, *args: str) -> str | None:
    result = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=False)
    return result.stdout.rstrip("\n") if result.returncode == 0 else None


def provenance(repo: Path) -> Provenance:
    """The state of the code in ``repo`` and of this process, read now."""
    status = _git(repo, "status", "--porcelain")
    return Provenance(
        commit=_git(repo, "rev-parse", "HEAD"),
        uncommitted=tuple(status.splitlines()) if status else (),
        command=tuple(sys.argv),
        started=datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        machine=f"{platform.system()} {platform.release()} {platform.machine()}, Python {platform.python_version()}",
    )
