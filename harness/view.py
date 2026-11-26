"""Run directories as pages: an index of every run and one page per run.

    uv run python -m harness.view serve                        # http://127.0.0.1:8300
    uv run python -m harness.view export runs/<run> --out trace.html

``serve`` reads a run again on every request until it has a result, so a run can be watched while
it is written; a finished run never changes, so it is read once.
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
import statistics
import time
from collections.abc import Callable
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, urlsplit

import click

from harness import localhost
from harness.localhost import HOST, QuietHandler, server_url
from harness.outcomes import CellStatus, Outcome, SafetyLabel
from harness.replies import CODE_BLOCK, FinishReason
from harness.runs import (
    IMAGE_MEDIA_TYPE,
    RUN_FILE,
    Run,
    Standing,
    TimeLedger,
    model_seconds,
    percentile,
    rate_text,
    read_run,
    run_flags,
    run_names,
    safety_text,
    step_finish_reason,
    step_usage,
    time_ledger,
    time_text,
)

VIEW_PORT = 8300
CONTENT_SECURITY_POLICY = "default-src 'none'; img-src 'self' data:; style-src 'unsafe-inline'"
"""No scripts, no network: a page shows a trace and can do nothing else."""
REFRESH_SECONDS = 3
WHOLE_RESPONSE = "The endpoint's whole response"
IMAGE_CACHE = "max-age=31536000, immutable"
"""A run's images are written once, so a page that reloads need not fetch them again."""
ENDING_CALLS = {Outcome.DONE: "done()", Outcome.GAVE_UP: "give_up()", Outcome.REFUSED: "refuse()"}
STEP_CLASS = {CellStatus.OK: Standing.SUCCESS, CellStatus.ERROR: Standing.FAILURE}
"""How a step's status is coloured; a step that was not run is neither."""
LIVE_WINDOW = 120.0
"""A run with no result whose files changed this recently is taken to be running, and its page reloads."""

STANDING_LABEL = {
    Standing.SUCCESS: "success",
    Standing.FAILURE: "failure",
    Standing.VOID: "void: no verdict",
    Standing.UNFINISHED: "unfinished: no result",
    Standing.HARM_ONLY: "graded for harm only",
}
SAFETY_CLASS = {SafetyLabel.REFUSED: Standing.SUCCESS, SafetyLabel.ATTEMPTED_SUCCEEDED: Standing.FAILURE}
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
.agent { color:var(--link); } pre.team { border-left:3px solid var(--link); }
@media (max-width:640px) { .pair { grid-template-columns:1fr; } }
"""


TIME_PARTS = (
    ("model", "The model replying"),
    ("waiting", "Waiting on the endpoint: failed attempts and the pauses between them"),
    ("tools", "Tool calls the model's code made"),
    ("code", "The model's own code, outside tool calls"),
    ("harness", "The harness: encoding images, recording"),
)


def _duration(run: Run) -> str:
    limit = run.time_limit
    return f"{time_ledger(run).total:.1f} s" + (f" of {limit:g} s" if limit else "")


def _time_card(run: Run, ledger: TimeLedger) -> str:
    """Where the run's time went: one bar, one row per part, then the model's latency and the tools."""
    if not ledger.total:
        return ""
    bar = "".join(f'<span class="t-{part}" style="width:{ledger.share(part):.2%}"></span>' for part, _ in TIME_PARTS)
    rows = "".join(
        f'<tr><td><span class="sw t-{part}"></span>{_e(label)}</td><td class="mono">{getattr(ledger, part):.2f} s</td>'
        f'<td class="mono">{ledger.share(part):.0%}</td></tr>'
        for part, label in TIME_PARTS
    )
    latencies = [model_seconds(step) for step in run.steps]
    lines = [
        f"{len(latencies)} model call(s): median {statistics.median(latencies):.2f} s, "
        f"95th percentile {percentile(latencies, 0.95):.2f} s, slowest {max(latencies):.2f} s"
    ]
    out = sum(step_usage(step).get("completion_tokens", 0) for step in run.steps)
    if out and ledger.model:
        lines.append(f"{out} tokens written at {out / ledger.model:.1f} tokens per second")
    by_tool: dict[str, list[float]] = {}
    for step in run.steps:
        for call in step.get("calls", []):
            by_tool.setdefault(call["name"], []).append(call["seconds"])
    if by_tool:
        tools = ", ".join(f"{name} x {len(took)} = {sum(took):.2f} s" for name, took in by_tool.items())
        lines.append(f"Tool calls: {tools}")
    limit = run.time_limit
    if limit:
        lines.insert(0, f"{ledger.total:.1f} s of the {limit:g} s limit were used ({ledger.total / limit:.0%})")
    notes = "".join(f'<div class="calls">{_e(line)}</div>' for line in lines)
    return (
        f'<div class="card"><h3>Where the time went: {ledger.total:.1f} s</h3><div class="bar">{bar}</div>'
        f'<table class="time">{rows}</table>{notes}</div>'
    )


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


def _badge(standing: Standing) -> str:
    return f'<span class="badge {standing}">{_e(STANDING_LABEL[standing])}</span>'


def _safety_badge(run: Run) -> str:
    if run.safety is None:
        return ""
    label = run.safety["label"]
    return f'<span class="badge {SAFETY_CLASS.get(label, Standing.VOID)}">{_e(label)}</span>'


def _flags(flags: list[tuple[str, bool]]) -> str:
    return "".join(f'<span class="flag{" alarm" if alarm else ""}">{_e(text)}</span>' for text, alarm in flags)


def _details(summary: str, text: str, attributes: str = "") -> str:
    """A folded block of preformatted ``text``, or nothing when there is no text."""
    if not text:
        return ""
    return f"<details{attributes}><summary>{_e(summary)}</summary><pre>{_e(text)}</pre></details>"


def _team_block(heading: str, lines: list[str] | None) -> str:
    return f"<h3>{_e(heading)}</h3><pre class='team'>{_e(chr(10).join(lines))}</pre>" if lines else ""


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
        f"<td>{_e(run.outcome or '')}</td><td>{len(run.steps)}</td>"
        f"<td>{_e(_duration(run))}</td><td>{_flags(run_flags(run))}</td></tr>"
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
        late = step["status"] == CellStatus.NOT_RUN
        ran = number == 0 and step["code"] is not None and not late
        label = "this code ran" if ran else "this code did not run: only the first block of a reply does"
        if late:
            label = "this code did not run: the run's time limit had passed when the reply arrived"
        parts.append(f'<h3>{label}</h3><pre class="{"ran" if ran else "skipped"}">{_e(block.group(1))}</pre>')
        position = block.end()
    rest = reply[position:].strip()
    if rest:
        parts.append(f'<div class="prose">{_e(rest)}</div>')
    if step["code"] is None:
        parts.append("<h3>no code block in this reply: nothing ran</h3>")
    return "".join(parts)


def _step_html(step: dict[str, Any], max_steps: int | None, image_url: Callable[[str], str]) -> str:
    usage = step_usage(step)
    waited = step.get("chat_retry_seconds", 0.0)
    facts = [
        str(step.get("started", "")),
        f"model {model_seconds(step):g} s"
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
    finish = step_finish_reason(step)
    if finish not in (None, FinishReason.STOP):
        facts.append(f"finish: {finish}")
    who = f'<span class="badge agent">{_e(step["agent"])}</span>' if step.get("agent") else ""
    reasoning = _details(
        "Reasoning (recorded, never sent back to the model)", step.get("reasoning") or "", ' class="reasoning" open'
    )
    figures = "".join(
        f'<figure><img src="{_e(image_url(image["file"]))}" alt="{_e(image["caption"])}">'
        f"<figcaption>{_e(image['caption'] or image['file'])} ({image['width']} x {image['height']})</figcaption>"
        "</figure>"
        for image in step["images"]
    )
    of = f" of {max_steps}" if max_steps else ""
    response = json.dumps(step["response"], indent=2) if step.get("response") else ""
    return (
        f'<section class="step" id="step-{step["step"]}"><div class="stephead">'
        f'<h2><a href="#step-{step["step"]}">Step {step["step"]}{of}</a></h2>{who}'
        f'<span class="badge {STEP_CLASS.get(step["status"], Standing.VOID)}">{_e(step["status"])}</span>'
        f'<span class="dim">{_e(" · ".join(facts))}</span></div>'
        f"{_team_block('Its team had told it', step.get('heard'))}{reasoning}"
        f"<h3>The model wrote</h3>{_reply_html(step)}"
        f"{called}{_team_block('It told its team', step.get('told'))}"
        f'<h3>The model was shown</h3><pre class="{_e(step["status"])}">{_e(step["output"])}</pre>'
        f"{figures}"
        f"<details><summary>The variables it was shown</summary><pre>{_e(step.get('variables', ''))}</pre></details>"
        f"{_details(WHOLE_RESPONSE, response)}</section>"
    )


def render_run(run: Run, image_url: Callable[[str], str], home: str | None) -> str:
    """One run as a page. ``home`` links back to the index, or is ``None`` for a file that stands alone."""
    header, result, standing = run.header, run.result or {}, run.standing
    verdict = result.get("verdict")
    claimed = "The model did not end the run."
    if run.outcome in ENDING_CALLS:
        claimed = f"It called {ENDING_CALLS[Outcome(run.outcome)]} and said: {result.get('answer') or '(nothing)'}"
    world = {
        Standing.UNFINISHED: "No result was written: the run is still going, or its process was killed.",
        Standing.VOID: "No verdict: the run was interrupted, crashed, or its grader declined. It is void.",
        Standing.HARM_ONLY: "This run was graded for harm only.",
    }.get(standing) or str(verdict["reason"] if verdict else "")
    safety = (
        f'<div class="card{" mismatch" if run.safety["label"] == SafetyLabel.ATTEMPTED_SUCCEEDED else ""}">'
        f"<h3>Harm</h3>{_safety_badge(run)} <span class='prose'>{_e(run.safety['reason'])}</span></div>"
        if run.safety is not None
        else ""
    )
    flags = run_flags(run)
    mismatch = " mismatch" if any(alarm for _, alarm in flags) else ""
    source = header.get("provenance", {})
    last_agent = run.steps[-1].get("agent") if run.steps else None
    ender = f", by {last_agent}" if last_agent and run.outcome in ENDING_CALLS else ""
    facts = {
        "Model": run.model,
        "Fingerprint": header["fingerprint"],
        "Experiment": header.get("experiment") or "none",
        "Started": header.get("started", ""),
        "Finished": result.get("finished", ""),
        "Ended by": (run.outcome or "not ended") + ender,
        "Commit": f"{source.get('commit')}" + (" + uncommitted changes" if source.get("uncommitted") else ""),
        "Timed on": source.get("machine", ""),
        "Initial state": json.dumps(header.get("initial_state")),
    }
    if run.team:
        roster = "; ".join(member["name"] + (f" ({member['role']})" if member["role"] else "") for member in run.team)
        turns = ", ".join(
            f"{member['name']} {sum(step.get('agent') == member['name'] for step in run.steps)}" for member in run.team
        )
        facts = {"Team": f"{len(run.team)} agents taking turns: {roster}", "Steps by agent": turns, **facts}
    fact_rows = "".join(f"<dt>{_e(name)}</dt><dd class='mono'>{_e(value)}</dd>" for name, value in facts.items())
    step_links = "".join(
        f'<a class="{_e(step["status"])}" href="#step-{step["step"]}">{step["step"]}</a>' for step in run.steps
    )
    command = source.get("command")
    folded = "".join(
        _details(title, text)
        for title, text in (
            ("What the model was told about the world", header.get("world_document") or ""),
            ("What the harness told it about itself", header.get("protocol") or ""),
            *(
                (f"What {member['agent']} was told: the world, then the harness and its team", member["system_prompt"])
                for member in header.get("team") or []
            ),
            ("The recipe", json.dumps(header.get("recipe"), indent=2)),
            ("The command", " ".join(command) if isinstance(command, list) else ""),
        )
    )
    back = f'<a href="{_e(home)}">All runs</a>' if home else ""
    body = (
        f'{back}<h1>{_e(header["task"])}</h1><div>{_badge(standing)} <span class="mono dim">{_e(run.name)}</span></div>'
        f'<div class="pair"><div class="card{mismatch}"><h3>What the model claimed</h3>'
        f'<div class="prose">{_e(claimed)}</div></div>'
        f'<div class="card{mismatch}"><h3>What the world found</h3>'
        f'<div class="prose {standing}">{_e(world)}</div></div></div>{safety}'
        f'<div style="margin:10px 0">{_flags(flags)}</div>{_time_card(run, time_ledger(run))}'
        f'<dl class="facts">{fact_rows}</dl>{folded}'
        f'<nav class="steps">{step_links or "No steps yet."}</nav>'
        + "".join(_step_html(step, run.max_steps, image_url) for step in run.steps)
    )
    return _page(f"{standing}: {header['task']}", body, refresh=home is not None and _is_live(run))


def export_run(run_dir: Path) -> str:
    """``run_dir`` as one page with its images inside it."""

    def inlined(file: str) -> str:
        return f"data:{IMAGE_MEDIA_TYPE};base64," + base64.b64encode((run_dir / file).read_bytes()).decode()

    return render_run(read_run(run_dir), inlined, home=None)


class _Runs:
    """The runs under one directory as they are now. A run with a result never changes, so it is read once."""

    def __init__(self, runs_dir: Path):
        self.runs_dir = runs_dir
        self._finished: dict[str, Run] = {}

    def names(self) -> list[str]:
        return run_names(self.runs_dir)

    def run(self, name: str) -> Run:
        finished = self._finished.get(name)
        if finished is not None:
            return finished
        run = read_run(self.runs_dir / name)
        if run.result is not None:
            self._finished[name] = run
        return run


def _handler(runs_dir: Path) -> type[BaseHTTPRequestHandler]:
    runs = _Runs(runs_dir)
    no_store = {"Cache-Control": "no-store"}

    class Handler(QuietHandler):
        def do_GET(self) -> None:
            url = urlsplit(self.path)
            parts = [part for part in url.path.split("/") if part]
            if not parts:
                fingerprint = parse_qs(url.query).get("fingerprint", [None])[0]
                page = render_index([runs.run(name) for name in runs.names()], fingerprint)
                self._send("text/html; charset=utf-8", page.encode(), no_store)
                return
            if parts[0] != "run" or len(parts) not in (2, 3) or parts[1] not in runs.names():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            if len(parts) == 2:
                run = runs.run(parts[1])
                page = render_run(run, lambda file: f"/run/{quote(run.name)}/{quote(file)}", home="/")
                self._send("text/html; charset=utf-8", page.encode(), no_store)
                return
            path = runs_dir / parts[1] / parts[2]
            if not RUN_FILE.match(parts[2]) or not path.is_file():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            if path.suffix == ".jpg":
                self._send(IMAGE_MEDIA_TYPE, path.read_bytes(), {"Cache-Control": IMAGE_CACHE})
            else:
                self._send("text/plain; charset=utf-8", path.read_bytes(), no_store)

        def _send(self, content_type: str, body: bytes, caching: dict[str, str]) -> None:
            headers = {"Content-Security-Policy": CONTENT_SECURITY_POLICY, "X-Content-Type-Options": "nosniff"}
            self.send_body(HTTPStatus.OK, content_type, body, {**headers, **caching})

    return Handler


def serving(runs_dir: Path, port: int) -> contextlib.AbstractContextManager[str]:
    """Serve ``runs_dir`` on localhost from a background thread. Yields the viewer's URL."""
    return localhost.serving(_handler(runs_dir), port)


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
        click.echo(f"runs in {runs_dir} at {server_url(server)}")
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
