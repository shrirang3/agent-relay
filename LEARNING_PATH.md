# agent-relay — Learning Path (you build, Claude co-pilots)

Build project for **Handoff Baton** (idea D): optimize *what context transfers when one agent hands off
to another*. You write the code + do the installs + run it + observe. Claude gives concept + doc pointers
+ skeletons-with-TODOs, then reviews your output. **No finished code handed over.**

Notes base (separate): `~/Agents/`. Plan: `../Agents/plans/handoff-baton.md` ·
Prereqs: `../Agents/resources/handoff-baton-prerequisites.md`

## Per-step loop
1. **Concept** (why) → 2. **Read** (exact doc) → 3. **Your task** (signatures + TODOs) →
4. **Run** (command + what to look for) → 5. Paste output → review → next.

## The ladder — feel the problem, then optimize
| # | Step | You'll understand | Runnable outcome |
|---|------|-------------------|------------------|
| 1 | **One agent + one tool** | the base ReAct loop | agent calls a tool, returns answer |
| 2 | **Traditional handoff (A→B, full-dump)** | why naive handoff is fragile | A plans, B finishes off dumped history |
| 3 | **Observe the pain** | token cost + where B loses the thread | measure baton size; spot a failure |
| 4 | **Structured baton (Pydantic)** | context as typed state, not transcript | A emits a typed baton; B runs off it |
| 5 | **Score it (DeepEval)** | objective pickup-success | TaskCompletion score per run |
| 6 | **Compare full-dump vs structured** | the core experiment; fairness | table: success vs tokens |
| 7 | **Redis as handoff channel** | the seam as a real, durable boundary | A→Redis Stream→B |
| 8 | **Cache + rate limiter (Redis)** | surviving the free tier | repeated calls skip Groq |
| 9 | **Context-cliff detector (ablation)** | *what* the handoff actually needs | Δsuccess ranking per field |

## Build log (you fill after each step)
> Record: what you ran, what you saw, what surprised you. Your revision trail.

### Step 1 —
- ran:
- saw:
- learned:

### Step 2 —
### Step 3 —
### Step 4 —
### Step 5 —
### Step 6 —
### Step 7 —
### Step 8 —
### Step 9 —
