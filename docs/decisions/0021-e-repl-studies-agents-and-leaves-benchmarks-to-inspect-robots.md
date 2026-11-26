---
date: 2026-11-26
status: adopted
summary: e_repl studies how an LLM agent behaves under a setup; benchmarking policies across robots is left to Inspect Robots
---

# 0021. e_repl studies agents and leaves benchmarks to Inspect Robots

## Decision

e_repl is a framework for studying language-model agents that control robots, for the questions
safety research asks of them: whether an agent declines harm or fails at it, whether its claims
match the world, whether it games its check, what contains the code it writes, whether a second
agent helps, how time pressure changes it, and whether its reasoning matches its actions. The
setup an agent is put in is the variable, and the run's trace is the data.

Benchmarking policies across tasks, robots, and simulators belongs to
[Inspect Robots](https://github.com/robocurve/inspect-robots)
([arXiv 2610.06306](https://arxiv.org/abs/2610.06306)). e_repl does not add VLA policies, an
embodiment registry, simulators, or benchmark suites. A feature is in scope when it lets a study
here vary the agent's setup or see more of what the agent did.

## Why

- Inspect Robots already does the benchmark job: tasks run as trials of a policy on an
  embodiment, scored into an eval log, with guards on each action. A second benchmark framework
  would split effort and results.
- Its LLM agents act through motion tools, one guarded chunk per call. An agent that writes code
  against a persistent namespace is a different subject: it composes calls, keeps state, and can
  do what no single tool call does. That is the case where the gate's place, the grader's reach,
  and the agent's claims matter most, and it is what decision 0001 built.
- Most of what e_repl records exists for agent-level questions and has no place in a benchmark
  score: separate endings for refusal and incapacity (0014, 0015), the model's claim beside the
  world's verdict (0005), each step's reasoning and whole response (0012), the time ledger
  (0016), and the messages between teammates (0020).
- Both use RoboHarm, so a harm task can be studied here and measured there.

## Alternatives

- Become a plugin of Inspect Robots. Its runs, records, and viewer would come for free; the
  persistent REPL, the one-action loop, and records that keep reasoning and claims would have to
  fit its trial and tool-call shapes.
- Grow into a general robotics eval framework: more embodiments, policies, and benchmarks here.
  It would repeat Inspect Robots and pull work away from the agent questions.
- Keep the template with no stated purpose. Nothing would decide which features belong.

## Cost if wrong

If Inspect Robots grows the agent-level records (code agents, reasoning traces, teams,
monitors), e_repl becomes a smaller duplicate and the case for a plugin gets stronger. If a
study here needs many tasks or several robots, it needs the bridge the README lists as not here
yet, or it has to move.
