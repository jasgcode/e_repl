"""The digest of the harness's own files (harness.digest)."""

from pathlib import Path

import pytest
from click.testing import CliRunner

from harness.digest import DIGEST_FILENAME, directory_digest, main


@pytest.fixture
def copy(tmp_path) -> Path:
    """A small stand-in for a project's copy of the harness."""
    path = tmp_path / "harness"
    path.mkdir()
    (path / "loop.py").write_text("x = 1\n")
    (path / "README.md").write_text("# harness\n")
    return path


def _check(copy: Path):
    return CliRunner().invoke(main, ["--harness", str(copy), "--check"])


def test_check_passes_on_a_recorded_copy():
    assert CliRunner().invoke(main, ["--check"]).exit_code == 0  # the repository's own harness


def test_an_edited_copy_fails_the_check_until_its_digest_is_rewritten(copy):
    assert CliRunner().invoke(main, ["--harness", str(copy)]).exit_code == 0
    assert _check(copy).exit_code == 0
    recorded = (copy / DIGEST_FILENAME).read_text()

    (copy / "loop.py").write_text("x = 2\n")
    failed = _check(copy)
    assert failed.exit_code != 0 and "python -m harness.digest" in failed.output
    assert (copy / DIGEST_FILENAME).read_text() == recorded  # --check writes nothing

    assert CliRunner().invoke(main, ["--harness", str(copy)]).exit_code == 0
    assert _check(copy).exit_code == 0 and (copy / DIGEST_FILENAME).read_text() != recorded


def test_a_copy_with_no_recorded_digest_fails_the_check(copy):
    assert _check(copy).exit_code != 0


def test_a_renamed_file_and_an_added_file_each_change_the_digest(copy):
    original = directory_digest(copy)
    (copy / "loop.py").rename(copy / "agent.py")
    renamed = directory_digest(copy)
    (copy / "extra.py").write_text("")
    assert len({original, renamed, directory_digest(copy)}) == 3


def test_caches_dot_files_and_the_digest_file_leave_the_digest_alone(copy):
    original = directory_digest(copy)
    (copy / "__pycache__").mkdir()
    (copy / "__pycache__" / "loop.cpython-313.pyc").write_bytes(b"\x00")
    (copy / ".DS_Store").write_bytes(b"\x00")
    (copy / DIGEST_FILENAME).write_text("sha256:stale\n")
    assert directory_digest(copy) == original
