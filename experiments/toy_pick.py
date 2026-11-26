"""Ask a model to pick the box of one color from the toy shelf.

    uv run python -m experiments.toy_pick red --model google/gemma-4-31b-it
    uv run python -m experiments.toy_pick red --model google/gemma-4-31b-it --world service

``--world service`` runs the shelf in its own process and gives the model a client of it.
``--time-limit SECONDS`` gives the run a wall-clock limit the model is told about, and
``--extra-latency SECONDS`` adds a delay to every reply, to see what speed is worth.

Needs an API key for the endpoint (``--api-key`` or ``NVIDIA_API_KEY``). The run is written to
``runs/``. Exits 0 when the shelf's verdict is a success.
"""

import contextlib
import logging
import sys
from enum import StrEnum
from functools import partial
from pathlib import Path

import click
import numpy as np

from harness.chat import NIM_BASE_URL, ChatModel, OpenAIChat, SlowedChat
from harness.docs import DOCS_DIR, OPEN, experiment_status
from harness.loop import AgentConfig, run_agent
from harness.repl import Repl, ReplConfig
from harness.trajectory import DirectoryRecorder
from project.shelf_client import ShelfClient
from project.shelf_service import running_shelf_service
from project.toy import COLORS, INSTRUCTIONS, pick_verdict, shuffled_shelf

REPO = Path(__file__).resolve().parents[1]


class World(StrEnum):
    IN_PROCESS = "in-process"
    """The shelf is an object in the harness's process."""
    SERVICE = "service"
    """The shelf is a process of its own, and the model holds a client of it."""


@click.command()
@click.argument("color", type=click.Choice(sorted(COLORS)))
@click.option("--model", required=True, help="Model ID at the endpoint, e.g. google/gemma-4-31b-it.")
@click.option("--base-url", default=NIM_BASE_URL, show_default=True, help="An OpenAI-compatible endpoint.")
@click.option("--api-key", envvar="NVIDIA_API_KEY", required=True, help="Defaults to $NVIDIA_API_KEY.")
@click.option("--seed", type=int, default=0, show_default=True, help="Orders the boxes on the shelf.")
@click.option(
    "--world",
    "world_name",
    type=click.Choice([world.value for world in World]),
    default=World.IN_PROCESS.value,
    show_default=True,
    help="Where the shelf runs: in the harness's process, or as a service the model holds a client of.",
)
@click.option("--max-steps", type=int, default=8, show_default=True)
@click.option("--time-limit", type=float, help="Seconds of wall clock the run may take. The model is told.")
@click.option("--extra-latency", type=float, default=0.0, show_default=True, help="Seconds added to every reply.")
@click.option("--experiment", help="Slug of the open docs/experiments file this run answers.")
@click.option("--runs", "runs_dir", type=click.Path(file_okay=False, path_type=Path), default=REPO / "runs")
def main(
    color: str,
    model: str,
    base_url: str,
    api_key: str,
    seed: int,
    world_name: str,
    max_steps: int,
    time_limit: float | None,
    extra_latency: float,
    experiment: str | None,
    runs_dir: Path,
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if experiment is not None and experiment_status(REPO / DOCS_DIR, experiment) != OPEN:
        raise click.ClickException(f"experiment {experiment!r} is closed; open a new one for new runs")
    world = World(world_name)
    chat: ChatModel = OpenAIChat(model=model, base_url=base_url, api_key=api_key)
    if extra_latency > 0:
        chat = SlowedChat(chat, extra_latency)
    recorder = DirectoryRecorder(runs_dir, repo=REPO)
    with contextlib.ExitStack() as stack:
        if world is World.SERVICE:
            service = stack.enter_context(running_shelf_service(seed, color))
            shelf, grade = ShelfClient(service.url), service.finish
        else:
            local = shuffled_shelf(seed)
            shelf, grade = local, partial(pick_verdict, local, color)
        result = run_agent(
            task=f"Pick the {color} box.",
            repl=Repl({"shelf": shelf, "np": np}, ReplConfig()),
            chat=chat,
            config=AgentConfig(instructions=INSTRUCTIONS, max_steps=max_steps, time_limit=time_limit),
            recorder=recorder,
            grade=grade,
            initial_state={"seed": seed, "world": world},
            judge_harm=None,
            experiment=experiment,
        )
    verdict = result.verdict
    assert verdict is not None  # both shelf graders always judge
    click.echo(f"{result.outcome}: {result.answer!r}; {verdict.reason}; {recorder.run_dir}")
    sys.exit(0 if verdict.success else 1)


if __name__ == "__main__":
    main()
