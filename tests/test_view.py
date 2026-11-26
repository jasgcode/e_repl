"""The trace viewer on real run directories: what a reader is shown, and what a model cannot do to the page."""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import requests

from harness import loop, repl
from harness.chat import Reply, ScriptedChat
from harness.loop import AgentConfig, run_agent
from harness.records import Harm, HarmFinding, Verdict
from harness.repl import Repl, ReplConfig
from harness.trajectory import STEPS_FILENAME, DirectoryRecorder
from harness.view import (
    ANY_FREE_PORT,
    CONTENT_SECURITY_POLICY,
    UNFINISHED,
    export_run,
    percentile,
    rate_text,
    read_run,
    read_runs,
    render_index,
    render_run,
    run_flags,
    safety_text,
    serving,
    time_ledger,
    time_text,
)
from tests.pacing import Clock, Paced

INSTRUCTIONS = "You switch a lamp."
YES, NO = Verdict(True, "the lamp is on"), Verdict(False, "the lamp is off")


def _cell(code: str) -> str:
    return f"Thinking aloud.\n```repl\n{code}\n```"


class Thinker:
    """A model that thinks before each scripted reply."""

    def __init__(self, replies: list[str], reasoning: str):
        self._chat, self._reasoning = ScriptedChat(replies), reasoning

    def recipe(self) -> dict:
        return {"model": "thinker"}

    def complete(self, messages) -> Reply:
        return Reply(self._chat.complete(messages).text, reasoning=self._reasoning)


def _run(runs: Path, name: str, chat, grade, max_steps: int = 4, judge_harm=None) -> Path:
    """Record one run and file it under ``runs/name``. Returns its directory."""
    recorder = DirectoryRecorder(runs / ".staging" / name, repo=runs)
    config = AgentConfig(instructions=INSTRUCTIONS, max_steps=max_steps)
    repl = Repl({"np": np}, ReplConfig())
    run_agent("Switch the lamp on.", repl, chat, config, recorder, grade, {"lamp": "off"}, judge_harm)
    shutil.move(recorder.run_dir, runs / name)
    return runs / name


def _page(run_dir: Path) -> str:
    return render_run(read_run(run_dir), lambda file: f"/run/{run_dir.name}/{file}", home="/")


def test_a_run_page_shows_what_the_model_thought_wrote_ran_and_was_shown(tmp_path):
    chat = Thinker(
        [_cell("show(np.zeros((8, 8), np.uint8), 'the lamp')\nlevel = 3"), _cell("done('it is on')")], "it looks off"
    )
    page = _page(_run(tmp_path, "a", chat, lambda: YES))
    assert "it looks off" in page and "never sent back to the model" in page
    assert "this code ran" in page and "level = 3" in page
    assert '<img src="/run/a/step001_0.jpg" alt="the lamp">' in page
    assert "level = 3" in page.split("The variables it was shown")[1]
    assert "It called done() and said: it is on" in page and "the lamp is on" in page
    assert '"lamp": "off"' in page.replace("&quot;", '"')


def test_nothing_a_model_writes_reaches_the_page_as_markup(tmp_path):
    hostile = (
        "<script>alert(1)</script>\n```repl\n"
        "print('<img src=x onerror=alert(2)>')\n"
        "show(np.zeros((8, 8), np.uint8), '\"><script>alert(3)</script>')\n"
        "done('<b>done</b>')\n```"
    )
    run_dir = _run(tmp_path, "a", Thinker([hostile], "</pre><script>alert(4)</script>"), lambda: NO)
    for page in (_page(run_dir), export_run(run_dir), render_index(read_runs(tmp_path), None)):
        assert "<script" not in page and "<b>done</b>" not in page and "<img src=x" not in page
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in _page(run_dir)
    assert CONTENT_SECURITY_POLICY in _page(run_dir) and "script-src" not in CONTENT_SECURITY_POLICY


def test_a_run_whose_ending_disagrees_with_the_worlds_verdict_is_an_alarm(tmp_path):
    def flags(name: str, cell: str, verdict: Verdict, max_steps: int = 4) -> list[tuple[str, bool]]:
        return run_flags(
            read_run(_run(tmp_path, name, ScriptedChat([_cell(cell)] * max_steps), lambda: verdict, max_steps))
        )

    assert flags("said-done-failed", "done('it is on')", NO) == [("said done, graded a failure", True)]
    assert flags("gave-up-succeeded", "give_up('cannot reach')", YES) == [("gave up, graded a success", True)]
    agree = flags("done", "done()", YES) + flags("gave-up", "give_up('cannot reach')", NO) + flags("ran-out", "1", NO, 2)
    assert not any(alarm for _, alarm in agree)
    assert 'class="card mismatch"' in _page(tmp_path / "said-done-failed")
    assert "It called give_up() and said: cannot reach" in _page(tmp_path / "gave-up")
    assert 'class="card mismatch"' not in _page(tmp_path / "gave-up")


def test_code_a_reply_held_that_did_not_run_is_marked_as_not_run(tmp_path):
    two_blocks = "First.\n```repl\nx = 1\n```\nThen.\n```repl\nlamp_off_forever = True\ndone()\n```"
    run_dir = _run(tmp_path, "a", ScriptedChat([two_blocks, "No block here.", _cell("done()")]), lambda: YES)
    page = _page(run_dir)
    ran, skipped = page.split("this code did not run")
    assert "x = 1" in ran and "lamp_off_forever = True" in skipped.split("The model was shown")[0]
    assert "no code block in this reply: nothing ran" in page
    texts = [text for text, _ in run_flags(read_run(run_dir))]
    assert "1 x reply with code that did not run" in texts and "1 x reply with no code" in texts


def test_the_index_counts_graded_runs_and_puts_void_ones_beside_the_rate(tmp_path):
    done = [_cell("done()")]
    _run(tmp_path, "1", ScriptedChat(done), lambda: YES)
    _run(tmp_path, "2", ScriptedChat(done), lambda: NO)
    _run(tmp_path, "3", ScriptedChat(done), lambda: None)
    _run(tmp_path, "4", ScriptedChat(done), lambda: YES, max_steps=9)  # another recipe, so another fingerprint
    runs = read_runs(tmp_path)
    assert [run.name for run in runs] == ["4", "3", "2", "1"]
    same = [run for run in runs if run.header["fingerprint"] == runs[1].header["fingerprint"]]
    assert rate_text(same) == "1 of 2 succeeded, 1 void"
    index = render_index(runs, None)
    assert "1 of 2 succeeded, 1 void" in index and "1 of 1 succeeded" in index
    only = render_index(runs, runs[0].header["fingerprint"])
    assert 'href="/run/4"' in only and 'href="/run/1"' not in only


def test_a_run_still_being_written_is_shown_with_the_steps_it_has(tmp_path):
    run_dir = _run(tmp_path, "a", ScriptedChat([_cell("x = 1"), _cell("done()")]), lambda: YES)
    (run_dir / "result.json").unlink()
    steps = (run_dir / STEPS_FILENAME).read_text().splitlines()
    (run_dir / STEPS_FILENAME).write_text(steps[0] + "\n" + steps[1][:40])  # the second line is half written
    run = read_run(run_dir)
    assert run.standing == UNFINISHED and [step["step"] for step in run.steps] == [1]
    assert 'http-equiv="refresh"' in _page(run_dir)
    assert 'http-equiv="refresh"' not in export_run(run_dir)


def test_an_exported_run_is_one_file_with_its_images_inside(tmp_path):
    chat = ScriptedChat([_cell("show(np.zeros((8, 8), np.uint8), 'the lamp')\ndone()")])
    page = export_run(_run(tmp_path, "a", chat, lambda: YES))
    assert '<img src="data:image/jpeg;base64,' in page
    assert "/run/" not in page and "All runs" not in page


def test_the_server_serves_pages_images_and_records_and_nothing_else_under_the_runs(tmp_path):
    chat = ScriptedChat([_cell("show(np.zeros((8, 8), np.uint8), 'the lamp')\ndone()")])
    run_dir = _run(tmp_path, "a", chat, lambda: YES)
    (run_dir / "notes.txt").write_text("not part of the record")
    (tmp_path / "secret.txt").write_text("outside every run")
    with serving(tmp_path, ANY_FREE_PORT) as url:
        index = requests.get(url, timeout=5)
        assert index.status_code == 200 and 'href="/run/a"' in index.text
        assert index.headers["Content-Security-Policy"] == CONTENT_SECURITY_POLICY
        assert "The model wrote" in requests.get(f"{url}/run/a", timeout=5).text
        assert requests.get(f"{url}/run/a/step001_0.jpg", timeout=5).content.startswith(b"\xff\xd8")
        assert json.loads(requests.get(f"{url}/run/a/run.json", timeout=5).text)["task"] == "Switch the lamp on."
        for path in ("/run/a/notes.txt", "/run/a/..%2Fsecret.txt", "/run/%2E%2E/secret.txt", "/run/b", "/secret.txt"):
            assert requests.get(url + path, timeout=5).status_code == 404


@pytest.mark.parametrize("runs", ["missing", "empty"])
def test_a_place_with_no_runs_says_so(tmp_path, runs):
    (tmp_path / "empty").mkdir()
    assert "No runs here yet" in render_index(read_runs(tmp_path / runs), None)


def test_safety_labels_are_counted_over_decided_runs_and_crossed_harm_is_an_alarm(tmp_path):
    def harm_run(name: str, cell: str, harm: Harm) -> Path:
        chat = ScriptedChat([_cell(cell)])
        return _run(tmp_path, name, chat, lambda: None, judge_harm=lambda: HarmFinding(harm, "seen on the table"))

    crossed = harm_run("1", "done('placed it')", Harm.CROSSED)
    refused = harm_run("2", "refuse('it would start a fire')", Harm.NONE)
    harm_run("3", "refuse('it would start a fire')", Harm.NONE)
    harm_run("4", "done('nothing to do')", Harm.NONE)  # no attempt and no refusal: invalid
    _run(tmp_path, "5", ScriptedChat([_cell("done()")]), lambda: None)  # no verdict, no harm task: void
    runs = read_runs(tmp_path)
    assert safety_text(runs) == "2 refused, 1 attempted_succeeded of 3 decided, 1 invalid"
    assert rate_text(runs) == "0 of 0 succeeded, 1 void"  # harm-only runs are not void
    assert ("crossed the harm threshold", True) in run_flags(read_run(crossed))
    assert not any(alarm for _, alarm in run_flags(read_run(refused)))
    page = _page(refused)
    assert "graded for harm only" in page and "It called refuse() and said: it would start a fire" in page
    assert "the model called refuse(); seen on the table" in page
    assert "2 refused, 1 attempted_succeeded of 3 decided, 1 invalid" in render_index(runs, None)


def test_a_runs_time_is_split_between_the_model_the_endpoint_the_tools_and_the_models_code(tmp_path, monkeypatch):
    clock = Clock()
    monkeypatch.setattr(loop, "time", clock)
    monkeypatch.setattr(repl, "time", clock)

    def move() -> None:
        clock.now += 4  # a tool that takes four seconds

    def think() -> None:
        clock.now += 1.5  # the model's own code, outside any tool

    chat = Paced(clock, [_cell("move()"), _cell("think_slowly()"), _cell("move()")], seconds=2, waits=1)
    recorder = DirectoryRecorder(tmp_path / ".staging", repo=tmp_path)
    config = AgentConfig(instructions=INSTRUCTIONS, max_steps=5, time_limit=18)
    tools = {"move": move}
    session = Repl(tools, ReplConfig(cell_timeout=None))
    session.namespace["think_slowly"] = think  # stands for code the model wrote itself
    run_agent("Switch the lamp on.", session, chat, config, recorder, lambda: NO, {}, None)
    run = read_run(recorder.run_dir)

    ledger = time_ledger(run)
    assert (ledger.model, ledger.waiting, ledger.tools, ledger.code, ledger.harness) == (6, 3, 8, 1.5, 0)
    assert ledger.total == 18.5 and round(ledger.share("tools"), 3) == round(8 / 18.5, 3)
    assert run.result is not None and run.result["outcome"] == "time_limit"
    assert ("ran out of time", False) in run_flags(run)
    page = render_run(run, lambda file: file, home="/")
    assert "Where the time went: 18.5 s" in page and "18.5 s of the 18 s limit were used (103%)" in page
    assert "3 model call(s): median 2.00 s, 95th percentile 2.00 s, slowest 2.00 s" in page
    assert "Tool calls: move x 2 = 8.00 s" in page and "then waited 1 s over 1 attempts" in page
    assert time_text([run]) == (
        "runs: median 18.5 s, slowest 18.5 s; the model replying was 32% of it; "
        "its latency over 3 calls: median 2.00 s, 95th percentile 2.00 s"
    )
    assert [call["at"] for step in run.steps for call in step["calls"]] == [3.0, 14.5]  # the two tool calls
    assert "18.5 s of 18 s" in render_index([run], None)


def test_a_run_that_mostly_waited_on_the_endpoint_is_flagged(tmp_path, monkeypatch):
    clock = Clock()
    monkeypatch.setattr(loop, "time", clock)
    chat = Paced(clock, [_cell("done()")], seconds=2, waits=30)
    run = read_run(_run(tmp_path, "a", chat, lambda: YES))
    assert ("94% of the run was waiting on the endpoint", False) in run_flags(run)


def test_a_reply_that_came_too_late_is_shown_as_code_that_did_not_run(tmp_path, monkeypatch):
    clock = Clock()
    monkeypatch.setattr(loop, "time", clock)
    chat = Paced(clock, [_cell("lamp_on = True")], seconds=30)
    recorder = DirectoryRecorder(tmp_path / ".staging", repo=tmp_path)
    config = AgentConfig(instructions=INSTRUCTIONS, max_steps=5, time_limit=20)
    run_agent("Switch the lamp on.", Repl({}, ReplConfig()), chat, config, recorder, lambda: NO, {}, None)
    page = render_run(read_run(recorder.run_dir), lambda file: file, home="/")
    assert "this code did not run: the run's time limit had passed when the reply arrived" in page
    assert "this code ran" not in page


def test_a_percentile_is_interpolated_between_the_samples_around_it():
    assert percentile([10, 20, 30, 40], 0.5) == 25
    assert percentile([10, 20, 30, 40], 0.9) == pytest.approx(37)
    assert percentile([7], 0.95) == 7
