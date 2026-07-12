# agent-relay — Problem & Tech Stack (plain-English)

The one-page "why this exists + what each piece does," explained with the office / shift-change analogy.
For the build steps see [../LEARNING_PATH.md](../LEARNING_PATH.md); for the full plan see
`~/Agents/plans/handoff-baton.md`.

---

## 1. The problem — a coworker shift change

You work a task, then go home. A coworker takes over and must finish it. You leave them a **sticky note**
so they can continue. That note is the **baton** — the only thing that crosses the handoff.

**Running example — a travel-booking assistant:**
- **Worker A (Planner)** chats with the user, figures out the trip.
- **Worker B (Booker)** has the tools to actually `search_flights()`, `book_flight()`, `book_hotel()`.

The user tells A a lot over many messages: *"Goa, 4 nights, Feb, budget $500, vegetarian, window seat,
NO red-eye, passport expires March 3."* Then A finishes planning and **hands off to B**. What A writes on
the note decides whether B succeeds.

### Three ways the handoff goes wrong
1. **Dump everything** — copy the whole 35-message chat onto the note (~4,000 tokens).
   Expensive every time, and the 3 critical constraints get **buried in noise** → B books a $620 red-eye.
   If the chat is too long, it gets cut off and a constraint **silently vanishes**.
2. **Lazy summary** — "User wants a Goa trip in Feb" (12 tokens).
   Cheap, but budget / seat / passport are **gone** → B books wrong.
3. **Tidy structured note** — labeled fields:
   `goal, constraints=[budget<=500, no red-eye, window, veg, passport 2026-03-03], decisions, open_steps`
   (~60 tokens). **Small AND complete** → B books correctly. ✅

**The problem in one line:** the model isn't dumb — **the handoff drops or buries what B needs.** Most
"agent B failed" bugs are really "the note was wrong." And nobody *measures* which lines the note truly
needs, so it's all guesswork and silent failures.

---

## 2. What's different from Claude Code (and LangGraph, CrewAI)

Every framework gives you a **pipe** — a way to spawn the next worker and pass a note. **None tell you
what to write on the note, or check whether it worked.**

### How Claude Code hands off (concrete)
When Claude Code spawns a subagent, the subagent's **prompt is the note.** Two modes:
```
fresh subagent  →  sees ONLY the note you write (prompt)  →  YOU pick what crosses  →  easy to forget stuff
fork subagent   →  inherits EVERYTHING (all context)      →  nothing lost           →  expensive / noisy
```
Plus the **return trip is filtered too** — you only get the subagent's final summary back, not its full
work. So the pipe squeezes context both ways.

Notice: "fresh" = the lazy-summary/structured-note case (you hand-write it, can forget things);
"fork" = the dump-everything case. **Same tradeoff as our travel example.** Claude Code gives you the
choice but still leaves the hard part — *what to put on the note* — to you.

### The gap = the product
| | Claude Code / LangGraph / CrewAI | agent-relay |
|---|---|---|
| Spawn next worker, isolate it, return result | ✅ built-in (the pipe) | uses the framework's pipe |
| Pass a note (prompt / fork) | ✅ but **you hand-write it** | ✅ **pluggable note styles** |
| Tell you *which* lines the note actually needs | ❌ | ⭐ **cliff detector** |
| Measure whether the coworker succeeded (+ cost) | ❌ | ⭐ **eval harness** |
| Works across any framework / model | ❌ (Claude-only, internal) | ✅ framework-agnostic |

**The pitch (positioning):**
> **agent-relay: find the smallest, complete note for handing off between agents — measured, not guessed.**
> - **smallest** = fewest tokens (cheap, fast)
> - **complete** = nothing the coworker needs is missing (doesn't fail)
> - **measured** = you *tested* the coworker succeeded, not "seems fine"

Not "a better pipe" (everyone has one). The unsolved part is **what goes in the note** — that's this project.
Complementary to Claude Code, not a competitor: it tells a CC/LangGraph user *what their subagent prompt
is missing* and **proves** it.

---

## 3. Tech stack — each piece as an office role

| Tech | Office analogy | Real job in agent-relay |
|---|---|---|
| **Python + `uv`** | the building + toolbox | the language; `uv` installs/manages tools |
| **LangGraph** | the **manager** running the shift change | defines Worker A → Worker B, decides *when* the handoff happens, passes control |
| **Groq (open-weight models)** | the workers' **brains** (free hired temps) | the LLM thinking inside A and B; free, no GPU |
| **Pydantic** | the **sticky-note template** (labeled boxes) | the `Baton` — typed fields: `goal`, `constraints`, `decisions`, `open_steps`. Forces a clean note |
| **Redis** | the **mailroom + cabinet + photocopier + bouncer + in-tray** | 5 jobs — see below; the heart of the handoff |
| **DeepEval + judge model** | the **QA inspector** | scores "did Worker B finish the job right?" Judge = a second, stronger brain grading the work |
| **Langfuse** | the **CCTV + logbook** | records what each worker did, time, tokens |
| **SQLite + DuckDB** | the **results spreadsheet** | tallies every run's scores for comparison |
| **Streamlit** | the **whiteboard** | shows the final comparison chart (success vs cost) |

### Redis's 5 jobs (the mailroom)
1. **Mailroom board** — A pins the note; B grabs it. *(the handoff itself)*
2. **Filing cabinet** — every note kept, so the inspector can pull an old one, erase a line, re-test.
   *(cliff detector — no need to re-run A)*
3. **Photocopier memory** — same question asked twice → hand back the saved copy, don't pay to re-think.
   *(cache — saves Groq quota)*
4. **Door bouncer** — limits how often workers call the brain, so the free agency (Groq) doesn't kick you out.
   *(rate limiter)*
5. **In-tray of jobs** — the pile of test tasks; many workers pull at once. *(work queue)*

---

## 4. How one run flows (all pieces together)
```
in-tray (Redis) → manager (LangGraph) starts Worker A (Groq brain)
  A fills the note template (Pydantic Baton)
  A pins note to mailroom board (Redis)          ← the handoff
Worker B (Groq brain) grabs note, does the job with its tools
  QA inspector (DeepEval + judge) scores: did B succeed?
  CCTV (Langfuse) logged time + tokens
  → score written to spreadsheet (SQLite) → shown on whiteboard (Streamlit)
```

**The experiment:** run this many times, changing **only the note style** (dump-everything vs tidy
template vs too-short). The spreadsheet shows which note wins on success + cost. Then the **cliff
detector** erases note fields one by one to find which lines the job can't live without.

### What we compare
| Metric | Plain meaning |
|---|---|
| **Pickup success** | did B finish the task right? (the point) |
| **Note size (tokens)** | how much the handoff costs each time |
| **Latency** | bigger note = slower start |
| **Tool correctness / steps** | did B act right after picking up? |
| **Robustness** | does the note survive long context without getting cut off? |

---

## TL;DR
The pipe (passing a note to the next agent) is solved by every framework, Claude Code included.
**What to put in the note — and proving it worked — is not.** agent-relay measures that: the smallest,
complete baton, and a cliff detector that shows exactly which context a handoff can't do without.
