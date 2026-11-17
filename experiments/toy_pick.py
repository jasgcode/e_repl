"""Ask a model to pick the box of one color from the toy shelf.

    uv run python -m experiments.toy_pick red --model google/gemma-4-31b-it

Needs an API key for the endpoint (``--api-key`` or ``NVIDIA_API_KEY``). The run is written to
``runs/``. Exits 0 when the model picked the right box.
"""

import logging
import sys
from pathlib import Path

import click
import numpy as np

from harness.chat import NIM_BASE_URL, OpenAIChat
from harness.docs import DOCS_DIR, OPEN, experiment_status
from harness.loop import AgentConfig, run_agent
from harness.repl import Repl, ReplConfig
from harness.trajectory import DirectoryRecorder
from project.toy import COLORS, INSTRUCTIONS, shuffled_shelf

REPO = Path(__file__).resolve().parents[1]


@click.command()
@click.argument("color", type=click.Choice(sorted(COLORS)))
@click.option("--model", required=True, help="Model ID at the endpoint, e.g. google/gemma-4-31b-it.")
@click.option("--base-url", default=NIM_BASE_URL, show_default=True, help="An OpenAI-compatible endpoint.")
@click.option("--api-key", envvar="NVIDIA_API_KEY", required=True, help="Defaults to $NVIDIA_API_KEY.")
@click.option("--seed", type=int, default=0, show_default=True, help="Orders the boxes on the shelf.")
@click.option("--max-steps", type=int, default=8, show_default=True)
@click.option("--experiment", help="Slug of the open docs/experiments file this run answers.")
@click.option("--runs", "runs_dir", type=click.Path(file_okay=False, path_type=Path), default=REPO / "runs")
def main(
    color: str,
    model: str,
    base_url: str,
    api_key: str,
    seed: int,
    max_steps: int,
    experiment: str | None,
    runs_dir: Path,
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if experiment is not None and experiment_status(REPO / DOCS_DIR, experiment) != OPEN:
        raise click.ClickException(f"experiment {experiment!r} is closed; open a new one for new runs")
    shelf = shuffled_shelf(seed)
    recorder = DirectoryRecorder(runs_dir, repo=REPO)
    result = run_agent(
        task=f"Pick the {color} box.",
        repl=Repl({"shelf": shelf, "np": np}, ReplConfig()),
        chat=OpenAIChat(model=model, base_url=base_url, api_key=api_key),
        config=AgentConfig(instructions=INSTRUCTIONS, max_steps=max_steps),
        recorder=recorder,
        experiment=experiment,
    )
    click.echo(f"{result.outcome}: {result.answer!r}; picked {shelf.picked}, wanted {color}; {recorder.run_dir}")
    sys.exit(0 if shelf.picked == color else 1)


if __name__ == "__main__":
    main()
