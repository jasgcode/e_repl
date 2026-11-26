"""Which parts of the harness can be used without the rest: the import graph, checked in a fresh interpreter."""

import subprocess
import sys

import pytest

AGENT_STACK = ("harness.chat", "harness.repl", "harness.loop", "harness.records", "numpy", "PIL")
"""What a grader, a hardware service, or a reader of runs should not have to load."""


@pytest.mark.parametrize(
    "module",
    [
        "harness.outcomes",
        "harness.replies",
        "harness.localhost",
        "harness.provenance",
        "harness.runs",
        "harness.operator",
        "harness.service",
        "harness.view",
    ],
)
def test_a_part_that_does_not_drive_a_model_loads_none_of_the_agent_stack(module):
    code = f"import sys, {module}; print([name for name in {AGENT_STACK!r} if name in sys.modules])"
    loaded = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert loaded.strip() == "[]"
