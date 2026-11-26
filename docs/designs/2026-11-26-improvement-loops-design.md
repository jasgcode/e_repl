---
date: 2026-11-26
summary: Campaigns of rounds where an agent's work carries over, a judge keeps or reverts it, and the world grades it held out
---

# Improvement loops

## Purpose

Robot self-improvement work runs one of a dozen loops: an agent revises a control program, grows
a skill library, picks its own practice, rewrites its own scaffold. Each loop has something the AI
changes and someone who decides the change worked. The failure reported for almost every one is
the same: the judge said yes and the world said no (a judge read asset files, test logs were
faked, a model graded its own homework, a library carried a mistake forward).

e_repl supports these loops as campaigns. A campaign records the judge's decision and, beside it,
the world's grade on conditions the judge never saw, so the gap between them is counted.

## The shape every loop shares

A campaign is put together from parts that swap independently, so one loop can run under several
judges and one judge under several loops, on any world:

| Part | What it is | Holds |
|---|---|---|
| `Loop` | a protocol: `kind`, `recipe()`, `task(index)`, `held_out(library)` | what each round asks the agent to change, and the world's grade of what it kept |
| `Judge` | a protocol: `kind`, `recipe()`, `judge(run, library)` | who decides a round's work is kept |
| practice | a function `(index, task) -> Practice` | the world and the model: one round as an ordinary run |
| `CampaignConfig` | `rounds`, `briefing` | how long, and what the next round is told |

`run_campaign` records each part's kind and recipe, taken from the objects it ran, so the record
cannot name one judge and run another. An experiment registers factories by kind and picks them
from flags (`--loop`, `--judge`).

A **campaign** is a sequence of rounds. In each round:

1. **Practice.** An ordinary run (`run_agent`, recorded as any run is). The agent holds a
   `Library`, a tool of named texts that outlives the round: a program, skills, notes.
2. **Judge.** The loop's own check decides whether this round's library is kept. Its kind is
   recorded: sealed tests, the agent's own claim, another model, code the agent wrote, a person.
3. **Held out.** The world grades the library as the judge saw it, on conditions the judge did
   not use. The judge never sees this grade.
4. **Keep or revert.** A rejected round's library is restored to what the last kept round left.
5. **Brief.** The next practice run's task says what the agent is told about this round.

Every round records the run's directory, how it ended, what the agent wrote, the judge's
acceptance, the held-out verdict, and whether the work was kept. `judge_agreement` counts the four
cells; *judge passed, world failed* is the number this design exists for.

What the next round is told is `CampaignConfig.briefing`, recorded with the campaign: nothing, the
judge's reason, or the judge's reason and the held-out grade. The last turns held-out conditions
into a training signal and is there to measure exactly that.

## The table, row by row

| Layer | Loop | Here | How |
|---|---|---|---|
| Behavior | Code-as-policy self-revision | yes, `program` | The library holds a program; the judge and held-out grade run it without the model. Toy: `experiments.toy_revise` |
| Behavior | Agent-as-policy | yes, already | `run_agent`: the model acts in the loop; its `done()` is the self-check, graded by the world |
| Behavior | Self-chosen practice | yes, `practice` | The library holds the tasks and limits the agent set itself; held out is a fixed task set it never chose |
| Training signal | LLM-written rewards | no | Needs a training loop to turn reward code into behavior |
| Training signal | Self-predicted reward | yes, as a judge | Judge kind `self`: the agent's own claim or estimate decides; `self_acceptance` is the claim version |
| Training signal | Real2Sim physics twin | partly, as a judge | Judge kind `agent_written`: code the agent wrote decides. A physics twin itself needs a simulator |
| Training signal | World-model dreaming | no | Needs a video world model |
| Learning procedure | RL from deployment | no | Updates weights |
| Learning procedure | Auto-research on robots | no | Fine-tunes models |
| Learning procedure | Harness and skill-library growth | yes, `skills` | The library holds functions carried into later, different tasks |
| Improver | Self-modifying agent code | yes, `scaffold` | The library holds notes the next round's agent is given; the judge is a set of runs |
| Improver | Weight self-editing | no | Edits weights |
| Body | Physical self-modification | no | Fabrication |
| Body | Production flywheel | no | A fleet |

The rows marked no belong to training frameworks. e_repl can host the agent half of some of
them later (an agent writing reward code that a small optimizer then uses), and that would be a
new decision.

## What is built now

- `harness/library.py`: `Library` (`write`, `read`, `names`), and `snapshot` and `restore`
  outside the tool, because only the campaign may revert.
- `harness/campaign.py`: the `Loop` and `Judge` protocols, `LoopKind`, `JudgeKind`, `Briefing`,
  `CampaignConfig`, `Acceptance`, `Round`, `run_campaign`, `judge_agreement`, `SelfJudge` (the
  one judge that needs no world), and recorders in memory and on disk
  (`campaigns/<time>-<loop>-<judge>/campaign.json`, `rounds.jsonl`, `end.json`).
- `src/project/toy_revision.py`: `ProgramLoop` and `SealedProgramJudge` on the shelf. The judge
  runs the program on shelves that expose their colors, as a simulator exposes its assets. The
  held-out grade runs it on shelves seen only through the camera, as a real one is. A program
  that reads `shelf.colors` passes the judge and fails the world.
- `experiments/toy_revise.py`: registries `LOOPS` and `JUDGES`, chosen with `--loop` and
  `--judge`. Adding a loop or a judge is one class and one registry line.
- Admission tests: a reference program is accepted and passes held out; a model that only calls
  `done()` is rejected; a program that reads the shelf's state is accepted and fails held out,
  and is counted as judge passed, world failed; a rejected round's library is restored; the same
  model and loop under `SelfJudge` keeps what the sealed judge rejected.

## Open

1. The `practice`, `skills`, and `scaffold` loops on the toy: each is a `Loop` class and a
   registry line. `skills` and `practice` vary the task by round, which `task(index)` allows;
   `scaffold` needs the next round's prompt to carry the notes, which practice functions do not
   yet take.
2. A `model` judge (another chat model reading the round against a constitution) and an
   `agent_written` judge (the agent's own checker from its library). Both fit `Judge` as it is,
   except that a `model` judge wants the transcript, which `Practice` does not carry yet.
3. Campaigns in the viewer.
4. Cost. Held out runs every round; on an arm that is the expensive part, and a campaign may want
   to grade every k rounds.
