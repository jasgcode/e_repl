"""Run an improvement loop on the toy shelf, with any loop and any judge registered here.

    uv run python -m experiments.toy_revise red --model google/gemma-4-31b-it --rounds 3
    uv run python -m experiments.toy_revise red --model google/gemma-4-31b-it --judge self

Each round is one run on a practice shelf, in which the model holds a library. ``--loop`` picks
what it is asked to change, ``--judge`` who decides the change is kept, and ``--briefing`` what
the next round is told. The world grades every round held out. Runs go to ``runs/`` and the
campaign's record to ``campaigns/``. To add a loop or a judge, write a class that fits
``harness.campaign.Loop`` or ``Judge`` and register a factory below.

Needs an API key for the endpoint (``--api-key`` or ``NVIDIA_API_KEY``).
"""

import logging
from collections.abc import Callable
from functools import partial
from pathlib import Path

import click
import numpy as np

from harness.campaign import (
    Briefing,
    CampaignConfig,
    DirectoryCampaignRecorder,
    Judge,
    JudgeKind,
    Loop,
    LoopKind,
    Practice,
    SelfJudge,
    judge_agreement,
    run_campaign,
)
from harness.chat import NIM_BASE_URL, ChatModel, OpenAIChat
from harness.docs import DOCS_DIR, OPEN, experiment_status
from harness.library import Library
from harness.loop import AgentConfig, run_agent
from harness.repl import Repl, ReplConfig
from harness.trajectory import DirectoryRecorder
from project.toy import COLORS, INSTRUCTIONS, pick_verdict, shuffled_shelf
from project.toy_revision import ProgramLoop, SealedProgramJudge

REPO = Path(__file__).resolve().parents[1]
JUDGE_SEEDS = (100, 101, 102)
HELD_OUT_SEEDS = (200, 201, 202, 203, 204)
"""Practice shelves are seeded by the round number, so the three sets never meet."""

LOOPS: dict[LoopKind, Callable[[str], Loop]] = {
    LoopKind.PROGRAM: lambda color: ProgramLoop(color, HELD_OUT_SEEDS),
}
JUDGES: dict[JudgeKind, Callable[[str], Judge]] = {
    JudgeKind.SEALED: lambda color: SealedProgramJudge(color, JUDGE_SEEDS),
    JudgeKind.SELF: lambda color: SelfJudge(),
}


@click.command()
@click.argument("color", type=click.Choice(sorted(COLORS)))
@click.option("--model", required=True, help="Model ID at the endpoint, e.g. google/gemma-4-31b-it.")
@click.option("--base-url", default=NIM_BASE_URL, show_default=True, help="An OpenAI-compatible endpoint.")
@click.option("--api-key", envvar="NVIDIA_API_KEY", required=True, help="Defaults to $NVIDIA_API_KEY.")
@click.option(
    "--loop",
    "loop_name",
    type=click.Choice([kind.value for kind in LOOPS]),
    default=LoopKind.PROGRAM.value,
    show_default=True,
)
@click.option(
    "--judge",
    "judge_name",
    type=click.Choice([kind.value for kind in JUDGES]),
    default=JudgeKind.SEALED.value,
    show_default=True,
)
@click.option("--rounds", type=click.IntRange(min=1), default=3, show_default=True)
@click.option(
    "--briefing",
    "briefing_name",
    type=click.Choice([briefing.value for briefing in Briefing]),
    default=Briefing.JUDGE.value,
    show_default=True,
    help="What the next round is told about the last.",
)
@click.option("--max-steps", type=int, default=8, show_default=True)
@click.option("--experiment", help="Slug of the open docs/experiments file this campaign answers.")
@click.option("--runs", "runs_dir", type=click.Path(file_okay=False, path_type=Path), default=REPO / "runs")
@click.option(
    "--campaigns", "campaigns_dir", type=click.Path(file_okay=False, path_type=Path), default=REPO / "campaigns"
)
def main(
    color: str,
    model: str,
    base_url: str,
    api_key: str,
    loop_name: str,
    judge_name: str,
    rounds: int,
    briefing_name: str,
    max_steps: int,
    experiment: str | None,
    runs_dir: Path,
    campaigns_dir: Path,
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if experiment is not None and experiment_status(REPO / DOCS_DIR, experiment) != OPEN:
        raise click.ClickException(f"experiment {experiment!r} is closed; open a new one for new runs")
    chat: ChatModel = OpenAIChat(model=model, base_url=base_url, api_key=api_key)
    library = Library()
    config = AgentConfig(instructions=INSTRUCTIONS, max_steps=max_steps)

    def practice(index: int, task: str) -> Practice:
        shelf = shuffled_shelf(index)
        recorder = DirectoryRecorder(runs_dir, repo=REPO)
        repl = Repl({"shelf": shelf, "library": library, "np": np}, ReplConfig())
        grade = partial(pick_verdict, shelf, color)
        result = run_agent(task, repl, chat, config, recorder, grade, {"seed": index}, None, experiment)
        return Practice(result, recorder.run_dir.name)

    recorder = DirectoryCampaignRecorder(campaigns_dir, repo=REPO)
    records = run_campaign(
        loop=LOOPS[LoopKind(loop_name)](color),
        judge=JUDGES[JudgeKind(judge_name)](color),
        config=CampaignConfig(rounds, Briefing(briefing_name)),
        library=library,
        practice=practice,
        recorder=recorder,
        experiment=experiment,
    )
    for record in records:
        accepted = "undecided" if record.acceptance is None else record.acceptance.reason
        held = "undecided" if record.held_out is None else record.held_out.reason
        click.echo(f"round {record.index}: kept={record.kept}; judge: {accepted}; held out: {held}")
    click.echo(f"{judge_agreement(records)}; {recorder.campaign_dir}")


if __name__ == "__main__":
    main()
