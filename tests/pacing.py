"""A clock a test moves by hand, and a scripted model that takes time on it."""

from harness.chat import Reply, ScriptedChat


class Clock:
    """Stands in for the ``time`` module of the loop and the REPL: nothing passes unless a test says so."""

    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now

    def time(self) -> float:
        return 1_800_000_000.0 + self.now


class Paced:
    """A scripted model: each reply takes ``seconds``, after ``waits`` seconds lost to the endpoint."""

    def __init__(self, clock: Clock, replies: list[str], seconds: float, waits: float = 0.0):
        self.clock, self.seconds, self.waits, self._chat = clock, seconds, waits, ScriptedChat(replies)

    def recipe(self) -> dict:
        return {"model": "paced", "seconds": self.seconds}

    @property
    def prompts(self) -> list:
        return self._chat.prompts

    def complete(self, messages) -> Reply:
        self.clock.now += self.seconds + self.waits
        text = self._chat.complete(messages).text
        return Reply(text, seconds=self.seconds + self.waits, request_seconds=self.seconds)
