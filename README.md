# agent-relay

**Find the smallest, complete baton for agent handoffs — measured, not guessed.**

![status](https://img.shields.io/badge/status-early%20WIP-orange) ![license](https://img.shields.io/badge/license-MIT-blue) ![python](https://img.shields.io/badge/python-3.12%2B-green)

> ⚠️ **Early / work-in-progress, built in the open.** The API below is the target design; not all of it
> is implemented yet. Follow the roadmap.

---

## What is this?

When one agent hands off to another (planner → executor, triage → specialist, a `fork`/subagent in
Claude Code, a LangGraph `Command`), it passes a **baton** — the context the next agent needs to continue.

Every framework gives you a **pipe** to pass that baton. **None tell you what to put in it, or whether
it worked.** So people either dump the entire conversation (expensive, noisy, truncates) or hand-write a
summary by gut feel (lossy). Handoffs become guesswork, and failures are silent — you can't tell if the
receiving agent failed because of a weak model or a bad baton.

**agent-relay** is a diagnostic + optimization toolkit for that seam:

- **Pluggable baton strategies** — `full_dump`, `summary`, `structured`, `minimal`.
- **Handoff eval harness** — score pickup success, token cost, and latency per strategy. Measured, not vibes.
- **Context-cliff detector** — mechanically find *which* baton fields are load-bearing by ablating each
  one and measuring the drop in success. Gives you a **minimal, complete** baton.

It's **framework-agnostic** (LangGraph first) and runs on **open-weight models** (via Groq / any
OpenAI-compatible endpoint). Complementary to Claude Code, LangGraph, CrewAI — it tells you what your
handoff is missing, and proves it.

## Why it matters

Multi-agent orchestration research keeps finding the same thing: **most agent failures are
context-transfer failures at handoff points, not model failures.** agent-relay makes that seam
measurable and tunable instead of guessed.

## Target API (roadmap)

```python
from agent_relay import Baton, HandoffEval, CliffDetector

# 1. pluggable baton strategies
baton = Baton.structured(from_agent=planner)      # or .full_dump(), .summary(), .minimal()

# 2. measure the handoff across strategies
report = HandoffEval(task_suite).compare(
    planner, executor, strategies=["full_dump", "structured"],
)
#   → success %, baton tokens, latency per strategy

# 3. the unique bit — what does the handoff actually NEED?
cliff = CliffDetector(executor).rank(baton, task_suite)
#   → {passport: CRITICAL (Δ-0.6), budget: CRITICAL, decisions: cut-it (Δ0.0)}
```

## Tech stack

LangGraph (handoff via `Command`) · open-weight models on Groq (`langchain-groq`) · Pydantic (typed
batons) · Redis (baton channel, cache, rate limiter, queue) · DeepEval (pickup-success scoring) ·
Langfuse (tracing) · SQLite/DuckDB (results). All free / self-hostable.

## Status & roadmap

| Stage | State |
|-------|-------|
| One agent + one tool (base loop) | 🔨 building |
| Traditional handoff (A→B, full-dump) | ⬜ |
| Structured baton (Pydantic) | ⬜ |
| Eval harness (pickup success) | ⬜ |
| Strategy comparison (Pareto: success vs tokens) | ⬜ |
| Redis handoff channel + cache + rate limiter | ⬜ |
| Context-cliff detector | ⬜ |

## Install (from source — not yet on PyPI)

```bash
git clone https://github.com/shrirang3/agent-relay
cd agent-relay
uv sync
cp .env.example .env   # add your GROQ_API_KEY
```

## Contributing

Early days — issues and ideas welcome. See `docs/` for the problem statement and architecture.

## License

MIT © Shrirang Mahankaliwar. Built in the open, co-developed with Claude.
