"""The REPL: what a cell's observation holds, and what survives between cells."""

import gc
import io
import signal
import weakref

import numpy as np
import pytest
from PIL import Image

from harness.images import ImageSettings
from harness.outcomes import CellStatus, Ending
from harness.repl import Repl, ReplConfig, describe_tools, truncate


def _repl(tools=None, config: ReplConfig | None = None) -> Repl:
    return Repl(tools or {}, config or ReplConfig())


def test_state_persists_and_a_trailing_expression_is_echoed():
    repl = _repl()
    repl.run("a = [1, 2, 3]")
    observation = repl.run("total = sum(a)\ntotal * 2")
    assert observation.status is CellStatus.OK
    assert observation.text.strip() == "12"
    assert "total = 6" in repl.variable_summary()


def test_an_error_shows_the_failing_line_and_the_session_survives():
    repl = _repl()
    repl.run("x = 1")
    observation = repl.run("y = 2\nz = x / 0")
    assert observation.status is CellStatus.ERROR
    assert "z = x / 0" in observation.text and "ZeroDivisionError" in observation.text
    assert repl.run("x + y").text.strip() == "3"  # y was assigned before the error


def test_a_tools_error_reaches_the_model_with_its_message():
    def move(x: float) -> None:
        raise ValueError(f"x={x} is out of reach: use x in [0.1, 0.4]")

    observation = _repl({"move": move}).run("move(0.9)")
    assert observation.status is CellStatus.ERROR
    assert "x=0.9 is out of reach: use x in [0.1, 0.4]" in observation.text


def test_a_syntax_error_is_an_observation():
    observation = _repl().run("def f(:\n  pass")
    assert observation.status is CellStatus.ERROR and "SyntaxError" in observation.text


def test_exit_does_not_end_the_session():
    repl = _repl()
    assert "call done()" in repl.run("x = 5\nexit()").text
    assert repl.run("x").text.strip() == "5"


def test_a_runaway_cell_is_interrupted():
    observation = _repl(config=ReplConfig(cell_timeout=0.2)).run("while True:\n    pass")
    assert observation.status is CellStatus.ERROR and "time limit" in observation.text


def test_an_alarm_set_outside_the_repl_is_still_pending_after_a_cell():
    signal.setitimer(signal.ITIMER_REAL, 30)
    try:
        _repl().run("x = 1")
        remaining, _ = signal.getitimer(signal.ITIMER_REAL)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    assert 0 < remaining <= 30


def test_truncation_keeps_the_head_and_the_tail():
    out = truncate("HEAD" + "x" * 10_000 + "TAIL", limit=1000)
    assert out.startswith("HEAD") and out.endswith("TAIL") and len(out) < 1200


def test_long_output_is_truncated_to_the_configured_limit():
    observation = _repl(config=ReplConfig(max_output_chars=200)).run("print('y' * 5000)")
    assert "chars truncated" in observation.text and len(observation.text) < 400


def test_done_ends_only_the_cell_that_called_it():
    repl = _repl()
    finished = repl.run("done('picked the red box')")
    assert finished.ending is Ending.DONE and finished.answer == "picked the red box"
    assert repl.run("1 + 1").ending is None


def test_each_tool_call_a_cell_makes_is_timed_once_under_the_name_the_model_used():
    class Arm:
        def goto(self, x: float) -> None:
            """Move."""

        def grasp(self, x: float) -> None:
            """Move twice."""
            self.goto(x)
            self.goto(x)

        def jam(self) -> None:
            """Fail."""
            raise RuntimeError("stuck")

    def weigh(parcel: str) -> float:
        """Weight in kilograms."""
        return 1.0

    arm = Arm()
    repl = _repl({"arm": arm, "weigh": weigh, "np": np})
    cell = "arm.grasp(1.0)\nweigh('a')\ntry:\n    arm.jam()\nexcept RuntimeError:\n    pass\nnp.zeros(2)"
    calls = repl.run(cell).calls
    assert [(call.name, call.raised) for call in calls] == [
        ("arm.grasp", None),  # its two calls to goto are part of it
        ("weigh", None),
        ("arm.jam", "RuntimeError"),
    ]
    assert all(call.seconds >= 0 for call in calls)
    assert repl.namespace["arm"] is arm and repl.run("type(arm).__name__").text == "'Arm'"
    assert repl.run("x = 1").calls == ()


def test_a_tool_can_report_facts_about_a_call_and_each_call_gets_its_own():
    class Camera:
        def __init__(self) -> None:
            self.call_facts: dict | None = None
            self.age = 0.4

        def frame(self) -> int:
            """A frame."""
            self.call_facts = {"frame_age": self.age}
            self.age += 0.1
            return 1

        def size(self) -> int:
            """The frame's size."""
            return 640

    camera = Camera()
    calls = _repl({"cam": camera}).run("cam.frame()\ncam.size()\ncam.frame()").calls
    assert [(call.name, call.facts) for call in calls] == [
        ("cam.frame", {"frame_age": 0.4}),
        ("cam.size", None),
        ("cam.frame", {"frame_age": 0.5}),
    ]
    assert camera.call_facts is None


def test_a_tool_whose_methods_cannot_be_timed_is_refused_when_the_repl_is_built():
    class Sealed:
        __slots__ = ()

        def read(self) -> int:
            """Read."""
            return 1

    with pytest.raises(TypeError, match="tool 'sealed' cannot be timed"):
        _repl({"sealed": Sealed()})


def test_refuse_ends_the_run_with_its_reason_and_needs_one():
    repl = _repl()
    stopped = repl.run("refuse('it would hurt someone')")
    assert stopped.ending is Ending.REFUSED and stopped.answer == "it would hurt someone"
    assert "refuse() needs a reason" in repl.run("refuse('')").text


def test_give_up_ends_the_run_with_its_reason_and_refuses_an_empty_one():
    repl = _repl()
    stopped = repl.run("give_up('the bin is out of reach')")
    assert stopped.ending is Ending.GAVE_UP and stopped.answer == "the bin is out of reach"
    silent = repl.run("give_up('  ')")
    assert silent.ending is None and silent.status is CellStatus.ERROR
    assert "give_up() needs a reason" in silent.text


def test_show_attaches_a_captioned_image():
    observation = _repl({"np": np}).run("show(np.zeros((4, 6), np.uint8), 'blank')")
    (image,) = observation.images
    assert (image.caption, image.width, image.height) == ("blank", 6, 4)


def test_show_refuses_float_pixels_and_says_how_to_fix_them():
    observation = _repl({"np": np}).run("show(np.zeros((4, 4)))")
    assert observation.status is CellStatus.ERROR and "uint8" in observation.text


def test_images_past_the_per_cell_limit_are_dropped_with_a_note():
    repl = _repl({"np": np}, ReplConfig(max_images_per_cell=2))
    observation = repl.run("for _ in range(3):\n    show(np.zeros((4, 4), np.uint8))")
    assert len(observation.images) == 2 and "image dropped" in observation.text


def test_a_shown_image_is_shrunk_and_keeps_its_colors():
    repl = _repl({"np": np}, ReplConfig(images=ImageSettings(max_side=64, quality=95)))
    observation = repl.run("red = np.zeros((100, 200, 3), np.uint8)\nred[..., 2] = 255\nshow(red)")
    (image,) = observation.images
    decoded = Image.open(io.BytesIO(image.jpeg)).convert("RGB")
    assert decoded.size == (64, 32) == (image.width, image.height)
    r, g, b = decoded.getpixel((32, 16))
    assert r > 240 and g < 15 and b < 15  # BGR red arrives as red


def test_a_tool_may_not_take_a_builtins_name():
    for name in ("show", "done", "give_up", "refuse"):
        with pytest.raises(ValueError, match=name):
            _repl({name: print})


def test_tools_are_described_by_signature_and_first_docstring_line():
    class Arm:
        """A two-finger arm."""

        def goto(self, x: float, speed: float = 0.05) -> None:
            """Move the tip to x metres.

            A second paragraph the model never sees.
            """

        def _private(self) -> None:
            """Hidden."""

    def ask(prompt: str) -> str:
        """Ask a sub-model."""
        return prompt

    described = describe_tools({"arm": Arm(), "ask": ask, "np": np})
    assert described.splitlines() == [
        "- arm: Arm. A two-finger arm.",
        "    arm.goto(x: float, speed: float = 0.05) -> None: Move the tip to x metres.",
        "- ask(prompt: str) -> str: Ask a sub-model.",
        "- np: module numpy",
    ]


def test_a_tool_held_by_several_repls_is_wrapped_once_and_keeps_none_of_them_alive():
    class Lamp:
        def switch(self) -> None:
            """Switch the lamp."""

    lamp = Lamp()
    first = _repl({"lamp": lamp})
    second = _repl({"light": lamp})
    assert lamp.switch.__wrapped__.__func__ is Lamp.switch  # not a wrapper around the first REPL's wrapper
    gone = weakref.ref(first)
    del first
    gc.collect()
    assert gone() is None
    assert [call.name for call in second.run("light.switch()").calls] == ["light.switch"]
