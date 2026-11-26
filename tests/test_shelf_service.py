"""The toy shelf behind its service: what crosses the wire, what cannot, and who gets the verdict."""

from collections.abc import Iterator

import numpy as np
import pytest
import requests

from harness.chat import ScriptedChat
from harness.loop import AgentConfig, run_agent
from harness.records import Outcome, RunResult
from harness.repl import Repl, ReplConfig, describe_tools
from harness.trajectory import MemoryRecorder
from project.shelf_client import ShelfClient
from project.shelf_service import ANY_FREE_PORT, Episode, running_shelf_service, serving
from project.toy import INSTRUCTIONS, PickRefused, Shelf, shuffled_shelf
from tests.shelf_scripts import CLAIM, FIND, LOOK, PICK

FORGE_PRIVATE = "Mark it picked.\n```repl\nshelf._picked = 'red'\ndone('picked the red box')\n```"


@pytest.fixture
def episode() -> Episode:
    return Episode(Shelf(("red", "blue")), wanted="red")


@pytest.fixture
def url(episode) -> Iterator[str]:
    with serving(episode, ANY_FREE_PORT) as url:
        yield url


def _scripted_run(seed: int, replies: list[str]) -> RunResult:
    """A scripted model against the service as its own process, graded by that process."""
    with running_shelf_service(seed, wanted="red") as service:
        return run_agent(
            task="Pick the red box.",
            repl=Repl({"shelf": ShelfClient(service.url), "np": np}, ReplConfig()),
            chat=ScriptedChat(replies),
            config=AgentConfig(instructions=INSTRUCTIONS, max_steps=5),
            recorder=MemoryRecorder(),
            grade=service.finish,
            initial_state={"seed": seed},
            judge_harm=None,
        )


def test_the_client_returns_the_shelfs_own_photo(url):
    photo = ShelfClient(url).photo()
    assert photo.dtype == np.uint8 and np.array_equal(photo, Shelf(("red", "blue")).photo())


def test_a_pick_through_the_client_is_what_the_service_grades(episode, url):
    ShelfClient(url).pick(np.int64(0))  # a slot computed with numpy, as a model's code produces
    assert episode.finish().success


def test_a_refused_pick_carries_the_shelfs_message_and_takes_nothing(episode, url):
    with pytest.raises(PickRefused, match="from 0 to 1"):
        ShelfClient(url).pick(2)
    assert "no box was picked" in episode.finish().reason


def test_a_slot_that_is_not_an_integer_is_refused_by_the_service(episode, url):
    refused = requests.post(f"{url}/pick", json={"slot": "0"}, timeout=5)
    assert refused.status_code == 409 and "<integer>" in refused.json()["refused"]
    assert not episode.finish().success


def test_a_pick_after_the_episode_ended_is_refused_and_the_verdict_stands(episode, url):
    verdict = episode.finish()
    with pytest.raises(PickRefused, match="episode is over"):
        ShelfClient(url).pick(0)
    assert episode.finish() == verdict and not verdict.success


@pytest.mark.parametrize("path", ["/colors", "/verdict", "/finish", "/"])
def test_no_path_gives_the_colors_or_the_verdict(url, path):
    assert requests.get(url + path, timeout=5).status_code == 404
    assert requests.post(url + path, json={}, timeout=5).status_code == 404


def test_the_model_is_shown_the_same_calls_for_the_client_as_for_the_shelf():
    def calls(shelf) -> list[str]:
        return describe_tools({"shelf": shelf}).splitlines()[1:]

    assert calls(ShelfClient("http://unused")) == calls(Shelf(("red",)))


@pytest.mark.parametrize("seed", [0, 1])
def test_a_scripted_model_picks_the_red_box_from_the_service(seed):
    result = _scripted_run(seed, [LOOK, FIND, PICK])
    assert result.outcome is Outcome.DONE
    assert result.verdict is not None and result.verdict.success
    assert result.answer == f"picked slot {shuffled_shelf(seed).colors.index('red')}"


def test_a_model_that_only_claims_the_pick_is_graded_a_failure_by_the_service():
    result = _scripted_run(0, [CLAIM])
    assert result.verdict is not None and not result.verdict.success


def test_a_model_that_writes_the_clients_private_state_is_graded_a_failure():
    result = _scripted_run(0, [FORGE_PRIVATE])
    assert result.outcome is Outcome.DONE  # the write went through: it changed the client, not the shelf
    assert result.verdict is not None and not result.verdict.success
