# agent-relay

**Find the smallest, complete baton for agent handoffs — measured, not guessed.**

![status](https://img.shields.io/badge/status-early%20WIP-orange) ![license](https://img.shields.io/badge/license-MIT-blue) ![python](https://img.shields.io/badge/python-3.13%2B-green)

> Built in the open. The API below is the target design — see [Status](#status) for what runs today.

---

## The problem

Whenever one agent hands work to another, something has to cross the seam between them:

- a **planner** hands a plan to an **executor**
- a **triage** agent routes to a **specialist**
- an **orchestrator** spawns a **subagent** and expects a result back
- a chain runs **A → B → C**, each hop compressing what the last one produced
- two agents from **different frameworks or vendors** exchange state over a protocol

Whatever crosses that seam is the **baton**.

Every framework gives you a pipe to carry it — a LangGraph `Command`, a state dict, a spawn prompt, an
A2A message. **None tell you what to put in it, or whether it worked.**

So teams do one of two things. Dump the whole transcript: expensive, noisy, and it truncates on long
runs. Or hand-write a summary by gut feel: lossy, and nobody knows which parts were load-bearing.
Either way the failures are silent — when the receiving agent does the wrong thing, you cannot tell
whether the model was weak or the baton was missing a field.

Multi-hop makes it worse. Each handoff compresses again, so a field dropped at the first seam is
unrecoverable at the third, and the failure surfaces far from where it was caused.

The reframe agent-relay is built on:

> **A handoff is lossy compression with a measurable downstream signal.**
> The receiving agent's task success is the loss function.

That turns baton design from a matter of taste into an empirical question.

## What it does

**Pluggable baton strategies.** `full_dump`, `summary`, `structured`, `minimal` — swap the compression
and hold everything else fixed.

**A handoff eval harness.** Pickup success and token cost per strategy, over K repeats. A Pareto view
of success versus size, so "cheaper" and "good enough" stay separate axes you can actually see.

**A context-cliff detector.** The headline. Ablate one baton field at a time, replay the receiver, and
rank fields by the drop in success. Large Δ means load-bearing. Δ0.0 means dead weight you can cut.
The output is a **minimal, complete** baton for your task — derived, not guessed.

```
goal            CRITICAL   Δ -0.80
budget          CRITICAL   Δ -0.60
open_steps      minor      Δ -0.05
decisions_log   cut it     Δ  0.00
```

## Method

For each field in the baton:

1. Freeze one extracted baton. Every trial runs against byte-identical text.
2. Remove the field — excluded from the payload entirely, not set to `null`.
3. **Verify the removal was real** (see below). If it wasn't, the field is reported as *untestable*.
4. Replay the receiver K times against the ablated baton. Only the receiver re-runs; the sending
   agent never runs again.
5. Δ = baseline success − ablated success.

## Why the numbers can be trusted

A tool that recommends cuts is only as good as its ability to tell a safe cut from one that merely
looked safe. Most of the design effort here goes into making a Δ mean something.

**Leak detection.** If `budget` is deleted from the payload but the goal line still reads "book a
flight under $500", the receiver succeeds and the field scores Δ0.0 — indistinguishable from a field
that genuinely doesn't matter. Same number, opposite conclusions. Before trusting any Δ, agent-relay
checks whether the removed fact survived elsewhere in the payload: by value, and by domain vocabulary
for facts that leak as wording rather than as a value. A leaking field is reported as **untestable**,
never as zero.

**Control fields.** Declare fields you *know* are noise. They must rank Δ0.0. If they don't, the
instrument is broken and the run says so — instead of you having to remember to sanity-check it.

This has already paid for itself. An early sweep reported the planted `small_talk` field as CRITICAL at
Δ−1.00, because trials killed by a provider rate limit were being counted as failed handoffs. A failed
HTTP call is **missing data, not a failed pickup**. Errored conditions are now reported `INVALID` with no
Δ at all, and a damaged baseline aborts the run rather than propagating into every row. Without a control
group, that false CRITICAL would have looked like a discovery.

**Missing data is never zero.** Empty field, leaking field, errored call — each is reported as
`UNTESTABLE` or `INVALID`. A harness that emits 0.0 for these is telling you to delete something it
never actually tested.

**A frozen baton.** Extraction is nondeterministic even at temperature 0. If the baton text drifts
between the baseline run and the ablated run, the Δ is measuring drift, not importance. One stored
baton per sweep.

**Tokens, not latency.** Token counts are exact and reproducible. Wall-clock on a shared inference
endpoint measures queue position — we have measured 0.5s and 151s for the same call minutes apart.

**Receiver isolation.** The receiving agent's system prompt carries tool hygiene only, never task
content. A prompt that says "satisfy all constraints" makes the receiver honour constraints the baton
never supplied, silently deflating every constraint field in the ranking.

## Target API

```python
from agent_relay import BatonBase, ControlField, CliffDetector, HandoffEval

class Baton(BatonBase):                      # your schema, your domain
    goal: str
    budget_usd: int | None = None
    user_mood: str = ControlField()          # declared noise; must rank Δ0.0

report = HandoffEval(tasks).compare(
    sender, receiver, strategies=["full_dump", "structured"],
)                                            # success %, baton tokens, per strategy

cliff = CliffDetector(receiver, success=lambda r: r.booked == "AI101").rank(baton, k=5)
#   → {budget_usd: CRITICAL (Δ-0.6), user_mood: cut-it (Δ0.0), goal: UNTESTABLE (leaks)}
```

The success function is **always yours**. It is the loss function, and no framework can infer it.

## Early results

All numbers below: synthetic planner→executor booking task, one frozen baton,
`llama-3.3-70b-versatile` at temperature 0.

### First cliff report

Seven baton fields, K=10 replays each, baseline 10/10:

| field | Δ | verdict | what the receiver did instead |
|---|---|---|---|
| `avoid_red_eye` | **−1.00** | CRITICAL | booked the $380 red-eye flight, 10/10 |
| `goal` | 0.00 | cut it | booked correctly with no goal at all |
| `budget_usd` | 0.00 | cut it | booked correctly with no budget |
| `superseded` | 0.00 | cut it | — |
| `open_steps` | — | UNTESTABLE | field came back empty; nothing to remove |
| `user_mood` *(control)* | — | pending | rate-limited; not yet measured |
| `small_talk` *(control)* | — | pending | rate-limited; not yet measured |

**The report is not yet validated.** The two control fields were lost to rate limiting, and until they
measure Δ0.00 the ranking above is provisional by this project's own standard.

Two of those zeros are findings about the **task**, not about batons — and worth stating because they're
the failure mode a naive ablation harness would report as fact:

- **`goal` is recoverable from the tool list.** The receiver holds a tool named `book_flight`. Remove
  the goal and it still books. Tool names are context you can't hide, so on this task the goal is
  genuinely redundant.
- **`budget_usd` is structurally redundant.** Once red-eye is excluded, the $420 flight is already the
  cheapest remaining option, so the $500 limit is never consulted. A constraint can only be measured if
  it is **independently decisive** — a property of the task fixture, invisible to the leak guard, and
  something that has to be designed out rather than detected.

### Strategy comparison

K=2 repeats per strategy:

| strategy | pickup success | baton tokens |
|---|---|---|
| `full_dump` | 2/2 | 758 |
| `structured` | 2/2 | **354** |

Same success for **2.1× less context**. Baton tokens are the receiver's real first-call input as
reported by the provider, not an estimate.

Read with the caveats, which matter more than the number:

- **K=2 is too small to separate success rates.** Both arms passed every trial, so this task currently
  discriminates on cost, not correctness.
- **Repeats in a tight loop are not independent samples.** At K=10 every condition came out identical
  token-for-token — yet the same condition disagreed between a K=3 run and a K=10 run made hours apart.
  Back-to-back replays hit the same backend state, so K inside a loop measures within-session
  determinism, not true variance. Real variance needs repeats spread over time.
- **One task, one model.** This demonstrates the harness works end to end. It is not a benchmark, and
  no general claim about baton strategies follows from it.

## Status

Working end to end on one synthetic A→B task — a planner handing a booking to an executor: typed baton
extraction from the sender's transcript, ablation by field exclusion, the two-layer leak guard with
self-correcting re-extraction, the K-repeat measurement loop, and the cliff sweep that produced the
report above. Sweeps are resumable per condition, so a rate limit costs one row rather than the run.

Immediately next: measure the two rate-limited control fields to validate the report, then redesign the
task fixture so each constraint is independently decisive — right now one constraint alone selects the
right answer, which makes the others unmeasurable however important they are.

Not built yet: the public `agent_relay` package, additional baton strategies, Redis as the handoff
channel, a judge-based scorer for tasks with no mechanical success check, and multi-hop (A→B→C) sweeps.

Current code lives in `sandbox/` as a deliberately concrete implementation. Extracting a package is
gated on two things: the cliff detector producing a real ranking, and a second, structurally different
task. One domain cannot validate a framework claim — it can only produce an abstraction shaped like
that domain.

## Install

```bash
git clone https://github.com/shrirang3/agent-relay
cd agent-relay
uv sync
cp .env.example .env   # add your GROQ_API_KEY
```

Runs on open-weight models via Groq, or any OpenAI-compatible endpoint. Framework-agnostic by design;
LangGraph is the first integration.

## Contributing

Early days — issues and ideas welcome, particularly from anyone who has debugged a bad handoff and had
no way to prove what was missing. See `docs/` for the problem statement and architecture.

## License

MIT © Shrirang Mahankaliwar. Built in the open, co-developed with Claude.
