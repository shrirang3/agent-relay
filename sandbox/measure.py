"""Step C — measure two baton strategies over K repeats.

One LLM run is a coin tossed once; you cannot tell "structured is better" from
"structured got lucky". This runs each strategy K times and reports pass rate, mean
input tokens, and WHICH flight the failures booked.

The failure id is the diagnosis:
    AI202 ($380, red-eye)   -> the red-eye constraint was lost
    AI303 ($610, no red-eye)-> the stale $600 budget survived the handoff
"""

import time
from collections import Counter

from dotenv import load_dotenv

from agents import make_executor, make_planner
from baton import LEAK_PROBES, get_baton, get_transcript
from fixtures import BOOKED, CORRECT_FLIGHT_ID, make_model, user_convo

K = 2

# Groq free tier allows ~30 req/min. A pickup costs ~3 model calls (decide to list,
# decide to book, final answer), so 4s between trials still ran ~67 calls/min and hit
# 429s mid-sweep. 8s keeps a 3-trial condition under the ceiling.
# The limit is per-minute and resets in about a minute — a 429 means slow down, not
# switch models. (openai/gpt-oss-120b was tried as a second bucket and is unusable
# here: it answers in prose without calling any tool.)
THROTTLE = 8.0

# Identical for both strategies, and deliberately free of task content. A trigger like
# "book the flight under $500" would hand B the task outside the baton — the same
# competing-channel problem as the receiver's system prompt.
TRIGGER = "Take over from here."


def run_trial(executor, messages) -> dict:
    """One handoff. `errored` trials count as failures but contribute no token data."""
    BOOKED.clear()  # mutate in place — fixtures holds this exact list object
    try:
        result = executor.invoke({"messages": messages})
    except Exception as e:
        # A dead trial is a failed trial, not a dead sweep. Tokens stay None so a
        # timeout can't drag the mean down and make a strategy look cheap.
        print(f"  ! {type(e).__name__}: {str(e)[:80]}")
        return {"ok": False, "flight": None, "total": None, "first": None, "errored": True}

    # First booking wins: if B books twice, the first call is the decision we're scoring.
    booked = BOOKED[0]["id"] if BOOKED else None

    # Real ingested tokens, not an estimate. Two numbers because they answer different
    # questions: `first` is roughly the baton itself, `total` is what the whole pickup
    # cost — B re-sends the growing conversation on every tool-loop turn, so `total`
    # also moves when a strategy causes more turns.
    per_call = [
        m.usage_metadata["input_tokens"]
        for m in result["messages"]
        if getattr(m, "usage_metadata", None)
    ]
    return {
        "ok": booked == CORRECT_FLIGHT_ID,
        "flight": booked,
        "total": sum(per_call),
        "first": per_call[0] if per_call else 0,
        "errored": False,
    }


def mean(xs) -> int:
    return round(sum(xs) / len(xs)) if xs else 0


def trials(executor, messages, k: int = K) -> dict:
    """Run the same handoff k times. Same input every time — the variance is B's."""
    runs = []
    for i in range(k):
        if i:
            time.sleep(THROTTLE)
        r = run_trial(executor, messages)
        runs.append(r)
        print(
            f"  trial {i + 1}/{k}: {'PASS' if r['ok'] else 'FAIL'}  "
            f"{r['flight'] or '-':<6} {r['first'] if r['first'] is not None else '-'} tok"
        )

    scored = [r for r in runs if not r["errored"]]  # token stats exclude dead calls
    return {
        "passes": sum(r["ok"] for r in runs),
        "k": k,
        "scored": len(scored),  # trials that produced data; k - scored == missing data
        "errors": sum(r["errored"] for r in runs),
        "first_tokens": mean([r["first"] for r in scored]),
        "total_tokens": mean([r["total"] for r in scored]),
        "booked": Counter(r["flight"] for r in runs),
    }


if __name__ == "__main__":
    load_dotenv()
    model = make_model()
    planner = make_planner(model)
    executor = make_executor(model)

    transcript = get_transcript(planner, user_convo)
    baton, attempts, leaks = get_baton(model, transcript, LEAK_PROBES)
    print(f"baton (attempts={attempts}, leaks={leaks or 'none'}):\n{baton.to_prompt()}\n")

    strategies = {
        "full_dump": list(transcript) + [("user", TRIGGER)],
        "structured": [("user", f"Handoff note:\n{baton.to_prompt()}\n\n{TRIGGER}")],
    }

    results = {}
    for name, messages in strategies.items():
        print(f"--- {name} ---")
        t = time.perf_counter()
        results[name] = trials(executor, messages)
        print(f"  {time.perf_counter() - t:.0f}s\n")

    # baton  = first-call input, roughly the note itself
    # total  = every call B made; also rises when a strategy causes more tool-loop turns
    print(f"{'strategy':<12} {'pass':>6} {'baton':>7} {'total':>7} {'err':>4}  failures")
    for name, r in results.items():
        fails = {f: n for f, n in r["booked"].items() if f != CORRECT_FLIGHT_ID}
        print(
            f"{name:<12} {r['passes']}/{r['k']:<4} {r['first_tokens']:>7} "
            f"{r['total_tokens']:>7} {r['errors']:>4}  {fails or '-'}"
        )