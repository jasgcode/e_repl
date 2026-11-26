"""Run directories as pages: an index of every run and one page per run.

    uv run python -m harness.view serve                        # http://127.0.0.1:8300
    uv run python -m harness.view export runs/<run> --out trace.html

``serve`` reads the directories on every request, so a run can be watched while it is written.
``export`` writes one run as a single file with its images inside, to send or attach.

The pages are for someone who was not there and wants to know what a model did: what it was
shown, what it thought, what it wrote, which code ran, what came back, and whether the world
agrees with what it claimed. Everything a model wrote is escaped and no page carries a script,
so a trace cannot act on the browser that reads it. Nothing is written into a run directory.
"""

import base64
import contextlib
import html
import json
import math
import re
import statistics
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, urlsplit

import click

from harness.loop import CODE_BLOCK
from harness.records import DECIDED_LABELS, SafetyLabel
from harness.trajectory import HEADER_FILENAME, RESULT_FILENAME, STEPS_FILENAME

HOST = "127.0.0.1"
VIEW_PORT = 8300
ANY_FREE_PORT = 0
SHUTDOWN_POLL = 0.02
CONTENT_SECURITY_POLICY = "default-src 'none'; img-src 'self' data:; style-src 'unsafe-inline'"
"""No scripts, no network: a page shows a trace and can do nothing else."""
RUN_FILE = re.compile(
    rf"^({re.escape(HEADER_FILENAME)}|{re.escape(STEPS_FILENAME)}|{re.escape(RESULT_FILENAME)}|step\d+_\d+\.jpg)$"
)
REFRESH_SECONDS = 3
ENDING_CALLS = {"done": "done()", "gave_up": "give_up()", "refused": "refuse()"}
STEP_CLASS = {"ok": "success", "error": "failure"}
"""How a step's status is coloured; a step that was not run is neither."""
LIVE_WINDOW = 120.0
"""A run with no result whose files changed this recently is taken to be running, and its page reloads."""

SUCCESS, FAILURE, VOID, UNFINISHED, HARM_ONLY = "success", "failure", "void", "unfinished", "harm-only"
STANDING_LABEL = {
    SUCCESS: "success",
    FAILURE: "failure",
    VOID: "void: no verdict",
    UNFINISHED: "unfinished: no result",
    HARM_ONLY: "graded for harm only",
}
SAFETY_CLASS = {SafetyLabel.REFUSED: SUCCESS, SafetyLabel.ATTEMPTED_SUCCEEDED: FAILURE}
"""How a safety label is coloured: a refusal as a success, crossed harm as a failure, the rest plainly."""

STYLE = """
:root { --bg:#fbfaf7; --fg:#1d1c1a; --dim:#6b6860; --line:#dedbd2; --card:#fff; --code:#f3f1ea;
  --ok:#1f7a4d; --bad:#b3261e; --void:#8a6d1d; --think:#f4efe1; --link:#1a4f8a; }
@media (prefers-color-scheme: dark) { :root { --bg:#16161a; --fg:#e7e5df; --dim:#9a968c; --line:#34343c;
  --card:#1e1e24; --code:#25252c; --ok:#5bc08a; --bad:#ff8a80; --void:#d9b95c; --think:#2a2720; --link:#8ab8f0; } }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--fg); font:15px/1.5 system-ui, sans-serif; }
main { max-width:1060px; margin:0 auto; padding:24px 16px 80px; }
a { color:var(--link); }
h1 { font-size:22px; margin:8px 0 12px; }
h2 { font-size:16px; margin:0; }
h3 { font-size:12px; text-transform:uppercase; letter-spacing:.06em; color:var(--dim); margin:14px 0 4px; }
pre { margin:0; padding:10px 12px; background:var(--code); border-radius:6px; overflow-x:auto;
  font:13px/1.45 ui-monospace, Menlo, monospace; white-space:pre-wrap; word-break:break-word; }
table { border-collapse:collapse; width:100%; font-size:14px; }
th, td { text-align:left; padding:6px 10px; border-bottom:1px solid var(--line); vertical-align:top; }
th { color:var(--dim); font-weight:600; font-size:12px; text-transform:uppercase; letter-spacing:.05em; }
.dim { color:var(--dim); }
.mono { font-family:ui-monospace, Menlo, monospace; font-size:13px; }
.badge { display:inline-block; padding:1px 9px; border-radius:99px; font-size:12px; font-weight:600;
  border:1px solid currentColor; white-space:nowrap; }
.success { color:var(--ok); } .failure { color:var(--bad); } .void, .unfinished, .harm-only { color:var(--void); }
.flag { display:inline-block; margin:2px 6px 2px 0; padding:1px 8px; border-radius:4px; font-size:12px;
  background:var(--code); border:1px solid var(--line); }
.flag.alarm { color:var(--bad); border-color:var(--bad); }
.pair { display:grid; grid-template-columns:1fr 1fr; gap:12px; margin:12px 0; }
.card { background:var(--card); border:1px solid var(--line); border-radius:8px; padding:12px 14px; }
.card.mismatch { border-color:var(--bad); }
.facts { display:grid; grid-template-columns:max-content 1fr; gap:2px 14px; font-size:14px; margin:12px 0; }
.facts dt { color:var(--dim); } .facts dd { margin:0; }
details { margin:8px 0; } summary { cursor:pointer; color:var(--dim); font-size:13px; }
nav.steps { position:sticky; top:0; background:var(--bg); padding:8px 0; border-bottom:1px solid var(--line);
  margin:16px 0; z-index:1; }
nav.steps a { display:inline-block; min-width:26px; text-align:center; margin:2px; padding:1px 4px; border-radius:4px;
  text-decoration:none; border:1px solid var(--line); font-size:13px; }
nav.steps a.error { color:var(--bad); border-color:var(--bad); }
section.step { background:var(--card); border:1px solid var(--line); border-radius:8px; padding:14px 16px;
  margin:14px 0; }
section.step { scroll-margin-top:56px; } section.step:target { outline:2px solid var(--link); }
.stephead { display:flex; flex-wrap:wrap; gap:8px 14px; align-items:baseline; }
.reasoning pre { background:var(--think); }
.prose { white-space:pre-wrap; margin:6px 0; }
.ran { border-left:3px solid var(--ok); } .skipped { border-left:3px solid var(--bad); opacity:.75; }
pre.error, pre.not_run { border-left:3px solid var(--bad); }
figure { margin:10px 0; } figure img { max-width:100%; border:1px solid var(--line); border-radius:6px; }
figcaption { font-size:13px; color:var(--dim); }
.bar { display:flex; height:14px; border-radius:4px; overflow:hidden; margin:8px 0; background:var(--code); }
.bar span { display:block; } .sw { display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:6px; }
.t-model { background:#5b7fd6; } .t-waiting { background:#c9a227; } .t-tools { background:#3f9d6b; }
.t-code { background:#8e6bbf; } .t-harness { background:#9a968c; }
table.time { width:auto; } table.time td { border:none; padding:1px 14px 1px 0; }
.calls { font-size:13px; color:var(--dim); margin:4px 0; }
@media (max-width:640px) { .pair { grid-template-columns:1fr; } }
"""


@dataclass(frozen=True)
class Run:
    """One run directory, read as it is now."""

    name: str
    header: dict[str, Any]
    steps: tuple[dict[str, Any], ...]
    result: dict[str, Any] | None
    """``None`` when ``result.json`` is not there: the run is being written, or its process was killed."""
    changed: float
    """Unix time the directory's files last changed."""

    @property
    def standing(self) -> str:
        if self.result is None:
            return UNFINISHED
        verdict = self.result.get("verdict")
        if verdict is None:
            return VOID if self.safety is None else HARM_ONLY
        return SUCCESS if verdict["success"] else FAILURE

    @property
    def safety(self) -> dict[str, str] | None:
        """The run's safety verdict (``label``, ``reason``), or ``None`` when its task defines no harm."""
        return (self.result or {}).get("safety")

    @property
    def model(self) -> str:
        return str(self.header.get("recipe", {}).get("chat", {}).get("model", ""))


def read_run(run_dir: Path) -> Run:
    lines = (run_dir / STEPS_FILENAME).read_text().splitlines() if (run_dir / STEPS_FILENAME).exists() else []
    steps = []
    for number, line in enumerate(lines, 1):
        try:
            steps.append(json.loads(line))
        except json.JSONDecodeError:
            if number < len(lines):
                raise
            # The last line of a run that is being written can be half there.
    result_path = run_dir / RESULT_FILENAME
    return Run(
        name=run_dir.name,
        header=json.loads((run_dir / HEADER_FILENAME).read_text()),
        steps=tuple(steps),
        result=json.loads(result_path.read_text()) if result_path.exists() else None,
        changed=max(path.stat().st_mtime for path in run_dir.iterdir()),
    )


def read_runs(runs_dir: Path) -> list[Run]:
    """Every run under ``runs_dir``, newest first. A directory without ``run.json`` is not a run."""
    if not runs_dir.is_dir():
        return []
    run_dirs = [path for path in runs_dir.iterdir() if (path / HEADER_FILENAME).is_file()]
    return [read_run(path) for path in sorted(run_dirs, key=lambda path: path.name, reverse=True)]


TIME_PARTS = (
    ("model", "The model replying"),
    ("waiting", "Waiting on the endpoint: failed attempts and the pauses between them"),
    ("tools", "Tool calls the model's code made"),
    ("code", "The model's own code, outside tool calls"),
    ("harness", "The harness: encoding images, recording"),
)
WAITING_WORTH_A_FLAG = 0.2
"""A run that spent more than this share of its time waiting on the endpoint says little about the model's speed."""


@dataclass(frozen=True)
class TimeLedger:
    """Where a run's wall time went, in seconds. The five parts add up to ``total``."""

    total: float
    model: float
    waiting: float
    tools: float
    code: float
    harness: float

    def share(self, part: str) -> float:
        return getattr(self, part) / self.total if self.total else 0.0


def time_ledger(run: Run) -> TimeLedger:
    """Split the time of ``run`` into the model, the endpoint, the tools, the model's code, and the harness."""
    model = sum(step.get("chat_request_seconds", step.get("chat_seconds", 0.0)) for step in run.steps)
    waiting = sum(step.get("chat_retry_seconds", 0.0) for step in run.steps)
    cells = sum(step.get("cell_seconds", 0.0) for step in run.steps)
    tools = sum(call["seconds"] for step in run.steps for call in step.get("calls", []))
    measured = model + waiting + cells
    total = max(run.steps[-1].get("elapsed", measured), measured) if run.steps else 0.0
    return TimeLedger(total, model, waiting, tools, max(cells - tools, 0.0), total - measured)


def percentile(values: list[float], share: float) -> float:
    """The value ``share`` of the way through ``values`` in order, interpolated between neighbours.

    Always taken over single samples. A percentile of per-run percentiles is not one.
    """
    ordered = sorted(values)
    position = (len(ordered) - 1) * share
    below = math.floor(position)
    above = min(below + 1, len(ordered) - 1)
    return ordered[below] + (ordered[above] - ordered[below]) * (position - below)


def _latencies(run: Run) -> list[float]:
    return [step.get("chat_request_seconds", step.get("chat_seconds", 0.0)) for step in run.steps]


def _time_limit(run: Run) -> float | None:
    return run.header.get("recipe", {}).get("agent", {}).get("time_limit")


def run_flags(run: Run) -> list[tuple[str, bool]]:
    """What a reader should look at first in ``run``, as (text, is an alarm) pairs.

    An alarm is a run that crossed its task's harm threshold, or one where the model's own account
    and the world's verdict disagree: it called ``done()`` and was graded a failure, or it called
    ``give_up()`` and was graded a success.
    """
    flags: list[tuple[str, bool]] = []
    if run.safety is not None and run.safety["label"] == SafetyLabel.ATTEMPTED_SUCCEEDED:
        flags.append(("crossed the harm threshold", True))
    ended_by = (run.result or {}).get("outcome")
    if ended_by == "done" and run.standing == FAILURE:
        flags.append(("said done, graded a failure", True))
    if ended_by == "gave_up" and run.standing == SUCCESS:
        flags.append(("gave up, graded a success", True))
    counts = {
        "cell error": sum(step["status"] == "error" and step["code"] is not None for step in run.steps),
        "reply with no code": sum(step["code"] is None for step in run.steps),
        "reply with code that did not run": sum(len(CODE_BLOCK.findall(step["reply"])) > 1 for step in run.steps),
        "reply cut off": sum(_finish_reason(step) == "length" for step in run.steps),
        "reply stopped by a content filter": sum(_finish_reason(step) == "content_filter" for step in run.steps),
    }
    flags += [(f"{count} x {text}", False) for text, count in counts.items() if count]
    if ended_by == "time_limit":
        flags.append(("ran out of time", False))
    ledger = time_ledger(run)
    if ledger.share("waiting") > WAITING_WORTH_A_FLAG:
        flags.append((f"{ledger.share('waiting'):.0%} of the run was waiting on the endpoint", False))
    if run.header.get("provenance", {}).get("uncommitted"):
        flags.append(("run from uncommitted code", False))
    return flags


def rate_text(runs: list[Run]) -> str:
    """The success rate of ``runs`` as the docs count it: graded runs only, void ones beside it."""
    graded = [run for run in runs if run.standing in (SUCCESS, FAILURE)]
    succeeded = sum(run.standing == SUCCESS for run in graded)
    void = sum(run.standing in (VOID, UNFINISHED) for run in runs)
    return f"{succeeded} of {len(graded)} succeeded" + (f", {void} void" if void else "")


def safety_text(runs: list[Run]) -> str:
    """The safety labels of ``runs`` as counts over decided runs, with invalid ones beside them."""
    labels = [run.safety["label"] for run in runs if run.safety is not None]
    decided = [label for label in labels if label in DECIDED_LABELS]
    if not labels:
        return ""
    counts = ", ".join(f"{decided.count(label)} {label}" for label in DECIDED_LABELS if label in decided)
    invalid = len(labels) - len(decided)
    return f"{counts or 'none'} of {len(decided)} decided" + (f", {invalid} invalid" if invalid else "")


def time_text(runs: list[Run]) -> str:
    """How long the finished runs of ``runs`` took, and how much of that was the model replying."""
    finished = [run for run in runs if run.result is not None and run.steps]
    if not finished:
        return ""
    ledgers = [time_ledger(run) for run in finished]
    totals = [ledger.total for ledger in ledgers]
    model_share = sum(ledger.model for ledger in ledgers) / (sum(totals) or 1.0)
    calls = [latency for run in finished for latency in _latencies(run)]  # pooled: every call is one sample
    return (
        f"runs: median {statistics.median(totals):.1f} s, slowest {max(totals):.1f} s; "
        f"the model replying was {model_share:.0%} of it; "
        f"its latency over {len(calls)} calls: median {statistics.median(calls):.2f} s, "
        f"95th percentile {percentile(calls, 0.95):.2f} s"
    )


def _duration(run: Run) -> str:
    limit = _time_limit(run)
    return f"{time_ledger(run).total:.1f} s" + (f" of {limit:g} s" if limit else "")


def _time_card(run: Run) -> str:
    """Where the run's time went: one bar, one row per part, then the model's latency and the tools."""
    ledger = time_ledger(run)
    if not ledger.total:
        return ""
    bar = "".join(f'<span class="t-{part}" style="width:{ledger.share(part):.2%}"></span>' for part, _ in TIME_PARTS)
    rows = "".join(
        f'<tr><td><span class="sw t-{part}"></span>{_e(label)}</td><td class="mono">{getattr(ledger, part):.2f} s</td>'
        f'<td class="mono">{ledger.share(part):.0%}</td></tr>'
        for part, label in TIME_PARTS
    )
    latencies = _latencies(run)
    lines = [
        f"{len(latencies)} model call(s): median {statistics.median(latencies):.2f} s, "
        f"95th percentile {percentile(latencies, 0.95):.2f} s, slowest {max(latencies):.2f} s"
    ]
    out = sum(((step.get("response") or {}).get("usage") or {}).get("completion_tokens", 0) for step in run.steps)
    if out and ledger.model:
        lines.append(f"{out} tokens written at {out / ledger.model:.1f} tokens per second")
    by_tool: dict[str, list[float]] = {}
    for step in run.steps:
        for call in step.get("calls", []):
            by_tool.setdefault(call["name"], []).append(call["seconds"])
    if by_tool:
        tools = ", ".join(f"{name} x {len(took)} = {sum(took):.2f} s" for name, took in by_tool.items())
        lines.append(f"Tool calls: {tools}")
    limit = _time_limit(run)
    if limit:
        lines.insert(0, f"{ledger.total:.1f} s of the {limit:g} s limit were used ({ledger.total / limit:.0%})")
    notes = "".join(f'<div class="calls">{_e(line)}</div>' for line in lines)
    return (
        f'<div class="card"><h3>Where the time went: {ledger.total:.1f} s</h3><div class="bar">{bar}</div>'
        f'<table class="time">{rows}</table>{notes}</div>'
    )


def _finish_reason(step: dict[str, Any]) -> str | None:
    choices = (step.get("response") or {}).get("choices") or [{}]
    return choices[0].get("finish_reason")


def _e(value: Any) -> str:
    return html.escape(str(value))


def _page(title: str, body: str, refresh: bool) -> str:
    reload = f'<meta http-equiv="refresh" content="{REFRESH_SECONDS}">' if refresh else ""
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        f'<meta http-equiv="Content-Security-Policy" content="{CONTENT_SECURITY_POLICY}">'
        f'<meta name="viewport" content="width=device-width, initial-scale=1">{reload}'
        f"<title>{_e(title)}</title><style>{STYLE}</style></head><body><main>{body}</main></body></html>"
    )


def _badge(standing: str) -> str:
    return f'<span class="badge {standing}">{_e(STANDING_LABEL[standing])}</span>'


def _safety_badge(run: Run) -> str:
    if run.safety is None:
        return ""
    return f'<span class="badge {SAFETY_CLASS.get(run.safety["label"], VOID)}">{_e(run.safety["label"])}</span>'


def _flags(run: Run) -> str:
    return "".join(f'<span class="flag{" alarm" if alarm else ""}">{_e(text)}</span>' for text, alarm in run_flags(run))


def _is_live(run: Run) -> bool:
    return run.result is None and time.time() - run.changed < LIVE_WINDOW


def render_index(runs: list[Run], fingerprint: str | None) -> str:
    """The page that lists runs, newest first, with one success rate per fingerprint."""
    shown = [run for run in runs if fingerprint is None or run.header["fingerprint"] == fingerprint]
    by_fingerprint: dict[str, list[Run]] = {}
    for run in runs:
        by_fingerprint.setdefault(run.header["fingerprint"], []).append(run)
    rates = "".join(
        f'<tr><td class="mono"><a href="/?fingerprint={quote(key)}">{_e(key)}</a></td><td>{_e(group[0].model)}</td>'
        f"<td>{_e(group[0].header.get('experiment') or '')}</td><td>{_e(rate_text(group))}</td>"
        f"<td>{_e(time_text(group))}</td>"
        f"<td>{_e(safety_text(group))}</td></tr>"
        for key, group in by_fingerprint.items()
    )
    rows = "".join(
        f'<tr><td class="mono"><a href="/run/{quote(run.name)}">{_e(run.name)}</a></td>'
        f"<td>{_e(run.header['task'])}</td><td>{_e(run.model)}</td><td>{_badge(run.standing)}</td>"
        f"<td>{_safety_badge(run)}</td>"
        f"<td>{_e((run.result or {}).get('outcome', ''))}</td><td>{len(run.steps)}</td>"
        f"<td>{_e(_duration(run))}</td><td>{_flags(run)}</td></tr>"
        for run in shown
    )
    filtered = (
        f'<p>Showing fingerprint <span class="mono">{_e(fingerprint)}</span>. <a href="/">Show all runs</a></p>'
        if fingerprint
        else ""
    )
    body = (
        "<h1>Runs</h1>"
        '<p class="dim">A rate counts graded runs only. A run with no verdict is void and is counted beside it.</p>'
        '<p class="dim">Safety labels are counted over decided runs: every label but invalid.</p>'
        "<table><tr><th>Fingerprint</th><th>Model</th><th>Experiment</th><th>Rate</th><th>Time</th><th>Safety</th></tr>"
        f"{rates}</table>"
        f"{filtered}<h3>{len(shown)} run(s), newest first</h3>"
        "<table><tr><th>Run</th><th>Task</th><th>Model</th><th>Verdict</th><th>Safety</th><th>Ended by</th>"
        "<th>Steps</th><th>Time</th>"
        f"<th>Look at</th></tr>{rows}</table>"
    )
    if not runs:
        body = '<h1>Runs</h1><p class="dim">No runs here yet. A run appears as soon as it starts.</p>'
    return _page("Runs", body, refresh=any(_is_live(run) for run in runs) or not runs)


def _reply_html(step: dict[str, Any]) -> str:
    """The model's reply as it wrote it: prose, then each code block marked as run or not run."""
    reply, parts, position = step["reply"], [], 0
    for number, block in enumerate(CODE_BLOCK.finditer(reply)):
        prose = reply[position : block.start()].strip()
        if prose:
            parts.append(f'<div class="prose">{_e(prose)}</div>')
        ran = number == 0 and step["code"] is not None and step["status"] != "not_run"
        label = "this code ran" if ran else "this code did not run: only the first block of a reply does"
        if step["status"] == "not_run":
            label = "this code did not run: the run's time limit had passed when the reply arrived"
        parts.append(f'<h3>{label}</h3><pre class="{"ran" if ran else "skipped"}">{_e(block.group(1))}</pre>')
        position = block.end()
    rest = reply[position:].strip()
    if rest:
        parts.append(f'<div class="prose">{_e(rest)}</div>')
    if step["code"] is None:
        parts.append("<h3>no code block in this reply: nothing ran</h3>")
    return "".join(parts)


def _step_html(run: Run, step: dict[str, Any], image_url: Callable[[str], str]) -> str:
    usage = (step.get("response") or {}).get("usage") or {}
    waited = step.get("chat_retry_seconds", 0.0)
    facts = [
        str(step.get("started", "")),
        f"model {step.get('chat_request_seconds', step.get('chat_seconds', 0)):g} s"
        + (f", then waited {waited:g} s over {step['chat_attempts']} attempts" if waited else ""),
        f"cell {step.get('cell_seconds', 0):g} s",
    ]
    if step.get("time_left") is not None:
        facts.append(f"{step['time_left']:.1f} s left")
    calls = " · ".join(
        f"{call['name']} {call['seconds']:.2f} s"
        + (f" raised {call['raised']}" if call["raised"] else "")
        + (f" ({', '.join(f'{name} {value}' for name, value in call['facts'].items())})" if call.get("facts") else "")
        for call in step.get("calls", [])
    )
    called = f'<div class="calls">Tool calls: {_e(calls)}</div>' if calls else ""
    if usage:
        facts.append(f"{usage.get('prompt_tokens', '?')} tokens in, {usage.get('completion_tokens', '?')} out")
    if _finish_reason(step) not in (None, "stop"):
        facts.append(f"finish: {_finish_reason(step)}")
    reasoning = (
        '<details class="reasoning" open><summary>Reasoning (recorded, never sent back to the model)</summary>'
        f"<pre>{_e(step['reasoning'])}</pre></details>"
        if step.get("reasoning")
        else ""
    )
    figures = "".join(
        f'<figure><img src="{_e(image_url(image["file"]))}" alt="{_e(image["caption"])}">'
        f"<figcaption>{_e(image['caption'] or image['file'])} ({image['width']} x {image['height']})</figcaption>"
        "</figure>"
        for image in step["images"]
    )
    budget = run.header.get("recipe", {}).get("agent", {}).get("max_steps")
    of = f" of {budget}" if budget else ""
    response = (
        f"<details><summary>The endpoint's whole response</summary><pre>{_e(json.dumps(step['response'], indent=2))}"
        "</pre></details>"
        if step.get("response")
        else ""
    )
    return (
        f'<section class="step" id="step-{step["step"]}"><div class="stephead">'
        f'<h2><a href="#step-{step["step"]}">Step {step["step"]}{of}</a></h2>'
        f'<span class="badge {STEP_CLASS.get(step["status"], VOID)}">{_e(step["status"])}</span>'
        f'<span class="dim">{_e(" · ".join(facts))}</span></div>'
        f"{reasoning}<h3>The model wrote</h3>{_reply_html(step)}"
        f"{called}"
        f'<h3>The model was shown</h3><pre class="{_e(step["status"])}">{_e(step["output"])}</pre>'
        f"{figures}"
        f"<details><summary>The variables it was shown</summary><pre>{_e(step.get('variables', ''))}</pre></details>"
        f"{response}</section>"
    )


def render_run(run: Run, image_url: Callable[[str], str], home: str | None) -> str:
    """One run as a page. ``home`` links back to the index, or is ``None`` for a file that stands alone."""
    header, result, standing = run.header, run.result, run.standing
    verdict = (result or {}).get("verdict")
    ended_by = (result or {}).get("outcome")
    claimed = "The model did not end the run."
    if ended_by in ENDING_CALLS:
        call = ENDING_CALLS[ended_by]
        claimed = f"It called {call} and said: {(result or {}).get('answer') or '(nothing)'}"
    world = {
        UNFINISHED: "No result was written: the run is still going, or its process was killed.",
        VOID: "No verdict: the run was interrupted, crashed, or its grader declined. It is void.",
        HARM_ONLY: "This run was graded for harm only.",
    }.get(standing) or str(verdict["reason"] if verdict else "")
    safety = (
        f'<div class="card{" mismatch" if run.safety["label"] == SafetyLabel.ATTEMPTED_SUCCEEDED else ""}">'
        f"<h3>Harm</h3>{_safety_badge(run)} <span class='prose'>{_e(run.safety['reason'])}</span></div>"
        if run.safety is not None
        else ""
    )
    mismatch = any(alarm for _, alarm in run_flags(run))
    source = header.get("provenance", {})
    facts = {
        "Model": run.model,
        "Fingerprint": header["fingerprint"],
        "Experiment": header.get("experiment") or "none",
        "Started": header.get("started", ""),
        "Finished": (result or {}).get("finished", ""),
        "Ended by": (result or {}).get("outcome", "not ended"),
        "Commit": f"{source.get('commit')}" + (" + uncommitted changes" if source.get("uncommitted") else ""),
        "Timed on": source.get("machine", ""),
        "Initial state": json.dumps(header.get("initial_state")),
    }
    fact_rows = "".join(f"<dt>{_e(name)}</dt><dd class='mono'>{_e(value)}</dd>" for name, value in facts.items())
    step_links = "".join(
        f'<a class="{_e(step["status"])}" href="#step-{step["step"]}">{step["step"]}</a>' for step in run.steps
    )
    folded = "".join(
        f"<details><summary>{_e(title)}</summary><pre>{_e(text)}</pre></details>"
        for title, text in (
            ("What the model was told about the world", header.get("world_document", "")),
            ("What the harness told it about itself", header.get("protocol", "")),
            ("The recipe", json.dumps(header.get("recipe"), indent=2)),
            ("The command", " ".join(source.get("command", [])) if isinstance(source.get("command"), list) else ""),
        )
        if text
    )
    back = f'<a href="{_e(home)}">All runs</a>' if home else ""
    body = (
        f'{back}<h1>{_e(header["task"])}</h1><div>{_badge(standing)} <span class="mono dim">{_e(run.name)}</span></div>'
        f'<div class="pair"><div class="card{" mismatch" if mismatch else ""}"><h3>What the model claimed</h3>'
        f'<div class="prose">{_e(claimed)}</div></div>'
        f'<div class="card{" mismatch" if mismatch else ""}"><h3>What the world found</h3>'
        f'<div class="prose {standing}">{_e(world)}</div></div></div>{safety}'
        f'<div style="margin:10px 0">{_flags(run)}</div>{_time_card(run)}<dl class="facts">{fact_rows}</dl>{folded}'
        f'<nav class="steps">{step_links or "No steps yet."}</nav>'
        + "".join(_step_html(run, step, image_url) for step in run.steps)
    )
    return _page(f"{run.standing}: {header['task']}", body, refresh=home is not None and _is_live(run))


def export_run(run_dir: Path) -> str:
    """``run_dir`` as one page with its images inside it."""

    def inlined(file: str) -> str:
        return "data:image/jpeg;base64," + base64.b64encode((run_dir / file).read_bytes()).decode()

    return render_run(read_run(run_dir), inlined, home=None)


def _handler(runs_dir: Path) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            url = urlsplit(self.path)
            parts = [part for part in url.path.split("/") if part]
            names = {path.name for path in runs_dir.iterdir()} if runs_dir.is_dir() else set()
            if not parts:
                fingerprint = parse_qs(url.query).get("fingerprint", [None])[0]
                self._send("text/html; charset=utf-8", render_index(read_runs(runs_dir), fingerprint).encode())
            elif parts[0] == "run" and len(parts) == 2 and parts[1] in names:
                run = read_run(runs_dir / parts[1])
                page = render_run(run, lambda file: f"/run/{quote(run.name)}/{quote(file)}", home="/")
                self._send("text/html; charset=utf-8", page.encode())
            elif parts[0] == "run" and len(parts) == 3 and parts[1] in names and RUN_FILE.match(parts[2]):
                path = runs_dir / parts[1] / parts[2]
                if not path.is_file():
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                kind = "image/jpeg" if path.suffix == ".jpg" else "text/plain; charset=utf-8"
                self._send(kind, path.read_bytes())
            else:
                self.send_error(HTTPStatus.NOT_FOUND)

        def _send(self, content_type: str, body: bytes) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Content-Security-Policy", CONTENT_SECURITY_POLICY)
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 (the base class's name)
            pass  # the base class writes every request to stderr

    return Handler


@contextlib.contextmanager
def serving(runs_dir: Path, port: int) -> Iterator[str]:
    """Serve ``runs_dir`` on localhost from a background thread. Yields the viewer's URL."""
    server = HTTPServer((HOST, port), _handler(runs_dir))
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": SHUTDOWN_POLL}, daemon=True)
    thread.start()
    try:
        yield f"http://{HOST}:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@click.group()
def main() -> None:
    """Read run directories as pages."""


@main.command()
@click.option(
    "--runs", "runs_dir", type=click.Path(file_okay=False, path_type=Path), default=Path("runs"), show_default=True
)
@click.option("--port", type=int, default=VIEW_PORT, show_default=True)
def serve(runs_dir: Path, port: int) -> None:
    """Serve every run under RUNS on localhost until Ctrl-C."""
    with HTTPServer((HOST, port), _handler(runs_dir)) as server:
        click.echo(f"runs in {runs_dir} at http://{HOST}:{server.server_port}")
        server.serve_forever()


@main.command()
@click.argument("run_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--out", type=click.Path(dir_okay=False, path_type=Path), required=True, help="The file to write.")
def export(run_dir: Path, out: Path) -> None:
    """Write one run as a single page with its images inside."""
    out.write_text(export_run(run_dir))
    click.echo(f"wrote {out}")


if __name__ == "__main__":
    main()
