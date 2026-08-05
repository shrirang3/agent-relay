"""Step D — the context-cliff detector.

For each field in a frozen baton: remove it, replay B K times, compare pass rate to the
baseline. A large drop means the field is load-bearing. Δ0.0 means dead weight.

A is never re-run. The baton is frozen on disk and only B replays, which is the whole
point of storing the baton (and, later, of Redis role 2).

Two gates run before any Groq call, because both produce a Δ0.0 that means nothing:
  - the field is empty        -> removing it removes nothing
  - the fact leaks elsewhere  -> removing the field doesn't remove the fact
Gated fields are reported UNTESTABLE, never as zero.
"""

import hashlib
import json
import pathlib
import time

from dotenv import load_dotenv

from agents import make_executor, make_planner
from baton import FIELDS, LEAK_PROBES, find_leaks, get_baton, get_transcript
from fixtures import CORRECT_FLIGHT_ID, make_model, user_convo
from measure import TRIGGER, trials

K = 3

# Per-condition results, so a 429 two thirds of the way through costs one condition
# instead of the whole sweep. Keyed by (baton text, K) — reusing rows measured against
# a different baton or a different K would silently mix incomparable numbers.
RESULTS_CACHE = pathlib.Path(__file__).parent / ".cliff.json"


def sweep_key(baton, k: int) -> str:
    return hashlib.sha256(f"{k}:{baton.to_prompt()}".encode()).hexdigest()[:12]


def load_results(key: str) -> dict:
    if RESULTS_CACHE.exists():
        d = json.loads(RESULTS_CACHE.read_text())
        if d.get("key") == key:
            return d
    return {"key": key, "baseline": None, "fields": {}}


def save_results(d: dict) -> None:
    RESULTS_CACHE.write_text(json.dumps(d, indent=2))


def keep(r: dict) -> dict:
    """The subset of a trials() result worth persisting."""
    return {
        "passes": r["passes"],
        "scored": r["scored"],
        "errors": r["errors"],
        "booked": {str(k): v for k, v in r["booked"].items()},
    }

# Fields planted as noise. They MUST rank Δ0.0 — if they don't, the instrument is broken
# and the whole ranking is suspect, so the run says so rather than relying on whoever
# reads the table to remember.
CONTROL_FIELDS = ("user_mood", "small_talk")


def note_messages(baton, drop=frozenset()):
    """B's input for one ablation. Identical shape to measure.py's structured arm."""
    return [("user", f"Handoff note:\n{baton.to_prompt(drop=drop)}\n\n{TRIGGER}")]


def verdict(delta: float) -> str:
    if delta <= -0.5:
        return "CRITICAL"
    if delta <= -0.2:
        return "important"
    if delta < 0:
        return "minor"
    if delta > 0:
        return "helps?"  # removing it improved things — noise, or the field misleads B
    return "cut it"


if __name__ == "__main__":
    load_dotenv()
    # More retries than the default: a sweep is dozens of calls against a free tier, and
    # one 429 must not turn into a missing row.
    model = make_model(max_retries=8)
    planner = make_planner(model)
    executor = make_executor(model)

    transcript = get_transcript(planner, user_convo)
    baton, attempts, leaks = get_baton(model, transcript, LEAK_PROBES)
    print(f"frozen baton (attempts={attempts}):\n{baton.to_prompt()}\n")
    if leaks:
        print(f"!! baton has leaks, those fields will be UNTESTABLE: {leaks}\n")

    store = load_results(sweep_key(baton, K))
    t0 = time.perf_counter()

    if store["baseline"] is None:
        print(f"--- baseline (complete baton), K={K} ---")
        base = keep(trials(executor, note_messages(baton), K))
        # Every delta is measured against this, so its errors would propagate into all
        # seven rows. Refuse to build a report on a damaged baseline.
        if base["errors"]:
            raise SystemExit(f"baseline had {base['errors']}/{K} errored calls — fix rate limits first")
        # A 0/K baseline has no success to lose, so every delta can only come out >= 0 —
        # the sweep is mathematically incapable of finding a cliff. It produced a table
        # anyway once, with "removing the budget helps, +1.00". Refuse instead.
        if base["passes"] == 0:
            raise SystemExit(
                f"baseline is 0/{K} — B fails with the COMPLETE baton, so no field can show a "
                f"drop. Fix the receiver or the task first. B booked: {base['booked']}"
            )
        if base["passes"] < K:
            raise SystemExit(
                f"baseline is {base['passes']}/{K} — B is at its decision boundary, so ablations "
                f"measure prompt perturbation rather than information. Make the task less "
                f"tempting before sweeping. B booked: {base['booked']}"
            )
        store["baseline"] = base
        save_results(store)
        print()
    else:
        print(f"baseline reused from cache: {store['baseline']['passes']}/{K}\n")
    base = store["baseline"]

    rows = []
    for f in FIELDS:
        value = getattr(baton, f)

        # gate 1 — nothing to remove. A bool is always testable; its probes do the work.
        if not isinstance(value, bool) and not value:
            rows.append((f, None, "UNTESTABLE", "field is empty"))
            continue

        # gate 2 — the fact survives the removal, so a Δ would be meaningless
        lk = find_leaks(baton, frozenset({f}), LEAK_PROBES)
        if lk:
            rows.append((f, None, "UNTESTABLE", lk[0]))
            continue

        if f in store["fields"]:
            r = store["fields"][f]
            print(f"--- drop {f} (cached) ---")
        else:
            print(f"--- drop {f} ---")
            r = keep(trials(executor, note_messages(baton, frozenset({f})), K))
            # Only cache clean conditions. Persisting an errored one meant a single 429
            # poisoned that row permanently — it replayed as cached INVALID on every
            # later run until the cache file was deleted by hand.
            if not r["errors"]:
                store["fields"][f] = r
                save_results(store)  # persist per condition, not at the end
            print()

        # A failed HTTP call is MISSING DATA, not a failed pickup. Scoring a 429 as a
        # failure once produced "small_talk is CRITICAL" — the exact false positive the
        # control fields exist to catch, manufactured by the scorer itself.
        if r["errors"]:
            rows.append((f, None, "INVALID", f"{r['errors']}/{K} calls errored — no data"))
            continue

        delta = (r["passes"] - base["passes"]) / K
        fails = {fl: n for fl, n in r["booked"].items() if fl != CORRECT_FLIGHT_ID}
        rows.append((f, delta, verdict(delta), fails or "-"))

    print(f"=== cliff report ===  baseline {base['passes']}/{K}  ({time.perf_counter() - t0:.0f}s)")
    print(f"{'field':<14} {'Δ':>6}  {'verdict':<11} what B booked instead")
    for f, delta, v, detail in rows:
        d = f"{delta:+.2f}" if delta is not None else "  -  "
        print(f"{f:<14} {d:>6}  {v:<11} {detail}")

    # instrument self-check
    bad = [
        (f, delta) for f, delta, v, _ in rows
        if f in CONTROL_FIELDS and delta is not None and delta != 0.0
    ]
    print()
    if bad:
        print(f"!! CONTROL FIELDS MOVED: {bad} — instrument suspect, do not trust this ranking")
    else:
        print("control fields flat at Δ0.0 — instrument behaving")