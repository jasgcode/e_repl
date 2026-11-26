"""A digest of the harness's own files, so a copy of it cannot change unnoticed.

    uv run python -m harness.digest            # rewrite harness/DIGEST
    uv run python -m harness.digest --check    # fail when the files no longer match it

A project's ``harness/`` is a copy of e_repl's. ``DIGEST`` names the exact files, and the test
suite recomputes it: an edit to a copy fails that project's tests until the digest is regenerated,
which puts the difference in a diff. Two copies with one digest hold the same files.
"""

import hashlib
from pathlib import Path

import click

DIGEST_FILENAME = "DIGEST"
DIGEST_PREFIX = "sha256:"
CACHE_DIRECTORY = "__pycache__"
HARNESS_DIR = Path(__file__).resolve().parent
LENGTH_BYTES = 8


def digested_files(directory: Path) -> list[str]:
    """The files a digest covers, as sorted ``/``-separated paths relative to ``directory``.

    Everything but the digest file, dot-files, and bytecode caches.
    """
    return sorted(
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file()
        and path != directory / DIGEST_FILENAME
        and not path.name.startswith(".")
        and CACHE_DIRECTORY not in path.parts
    )


def directory_digest(directory: Path) -> str:
    """``sha256:<hex>`` over each covered file's relative path and bytes, in path order."""
    sha = hashlib.sha256()
    for relative in digested_files(directory):
        for part in (relative.encode(), (directory / relative).read_bytes()):
            # Length first, so no two sets of files can share one byte stream.
            sha.update(len(part).to_bytes(LENGTH_BYTES, "big"))
            sha.update(part)
    return DIGEST_PREFIX + sha.hexdigest()


@click.command()
@click.option("--check", is_flag=True, help="Write nothing; fail when the files do not match the recorded digest.")
@click.option(
    "--harness",
    "harness_dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=HARNESS_DIR,
    help="The harness directory. Defaults to the one this module is in.",
)
def main(check: bool, harness_dir: Path) -> None:
    """Rewrite the harness's DIGEST from its files."""
    digest = directory_digest(harness_dir)
    path = harness_dir / DIGEST_FILENAME
    if not check:
        path.write_text(f"{digest}\n")
        return
    if not path.is_file() or path.read_text().strip() != digest:
        raise click.ClickException(
            f"{harness_dir} no longer matches {path.name}. If the change is meant, run: uv run python -m harness.digest"
        )


if __name__ == "__main__":
    main()
