"""Step D — the context-cliff detector.

For each field in a frozen baton: remove it, replay B K times, compare pass rate to the
baseline. A large drop means the field is load-bearing. Zero means dead weight.

A is never re-run. The baton is frozen on disk and only B replays.

All trustworthiness logic lives in validity.py as pure functions. This file measures;
validity.py decides whether a measurement means anything.
"""

import hashlib
import json
import pathlib
import time

from dotenv import load_dotenv

import validity as V
from agents import make_executor, make_planner
from baton import FIELDS, LEAK_PROBES, find_leaks, get_baton, get_transcript
from fixtures import CORRECT_FLIGHT_ID, FLIGHTS, make_model, user_convo
from measure import TRIGGER, trials

K = 3

CONTROL_FIELDS = ("user_mood", "small_talk")

RESULTS_CACHE = pathlib.Path(__file__).parent / ".cliff.json"


def sweep_key(baton, k: int) -> str:
    """Identity of a sweep. Includes the FLIGHT TABLE, not just the baton — rows
    measured against a different task are not comparable, and the cache would have
    served them silently when the fixture changed."""
    blob = f"{k}:{baton.to_prompt()}:{FLIGHTS!r}"
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def load_results(key: str) -> dict:
    if RESULTS_CACHE.exists():
        d = json.loads(RESULTS_CACHE.read_text())
        if d.get("key") == key:
            return d
    return {"key": key, "floor": None, "baseline": None, "fields": {}}


def save_results(d: dict) -> None:
    RESULTS_CACHE.write_text(json.dumps(d, indent=2))


def keep(r: dict) -> dict:
    return {
        "passes": r["passes"],
        "scored": r["scored"],
        "errors": r["errors"],
        "booked": {str(k): v for k, v in r["booked"].items()},
    }


def note_messages(baton, drop=frozenset()):
    """B's input for one condition. Dropping every field yields `{}` — the message
    shape stays identical, only the content empties out."""
    return [("user", f"Handoff note:\n{baton.to_prompt(drop=drop)}\n\n{TRIGGER}")]


def verdict(imp: float | None) -> str:
    """Graded on normalised importance, not raw delta — raw deltas are in units of
    'trials out of K' and are not comparable across tasks."""
    if imp is None:
        return "?"
    if imp >= 0.99:
        return "CRITICAL"
    if imp >= 0.5:
        return "important"
    if imp > 0:
        return "minor"
    if imp < 0:
        return "misleads?"   # worse than no note at all
    return "cut it"


def measure(executor, store, name, baton, drop, k):
    """Run one condition, or replay it from cache. Only clean runs are cached — a
    persisted 429 poisoned its row on every later run until the file was deleted."""
    if store["fields"].get(name):
        print(f"--- {name} (cached) ---")
        return store["fields"][name]
    print(f"--- {name} ---")
    r = keep(trials(executor, note_messages(baton, drop), k))
    if not r["errors"]:
        store["fields"][name] = r
        save_results(store)
    print()
    return r


if __name__ == "__main__":
    load_dotenv()
    model = make_model(max_retries=8)
    planner = make_planner(model)
    executor = make_executor(model)

    transcript = get_transcript(planner, user_convo)
    baton, attempts, leaks = get_baton(model, transcript, LEAK_PROBES)
    print(f"frozen baton (attempts={attempts}):\n{baton.to_prompt()}\n")

    store = load_results(sweep_key(baton, K))
    t0 = time.perf_counter()

    # --- G0: floor. Empty baton must FAIL, or nothing downstream can register. ---
    if store["floor"] is None:
        print(f"--- floor (EMPTY baton), K={K} ---")
        store["floor"] = keep(trials(executor, note_messages(baton, frozenset(FIELDS)), K))
        save_results(store)
        print()
    floor = store["floor"]
    if floor["errors"]:
        raise SystemExit(f"floor had {floor['errors']}/{K} errored calls — fix rate limits first")
    v = V.check_floor(floor["passes"], K)
    if not v.ok:
        raise SystemExit(f"{v.code}: {v.reason}\n  B booked without a note: {floor['booked']}")
    print(f"floor {floor['passes']}/{K} — the note carries real signal\n")

    # --- G1: baseline. Full baton must SUCCEED every time. ---
    if store["baseline"] is None:
        print(f"--- baseline (complete baton), K={K} ---")
        store["baseline"] = keep(trials(executor, note_messages(baton), K))
        save_results(store)
        print()
    base = store["baseline"]
    v = V.check_baseline(base["passes"], base["errors"], K)
    if not v.ok:
        raise SystemExit(f"{v.code}: {v.reason}\n  B booked: {base['booked']}")

    # --- per-field sweep ---
    rows, deltas = [], {}
    for f in FIELDS:
        # G2/G3 — free, so they run before any API call
        v = V.check_field(getattr(baton, f), find_leaks(baton, frozenset({f}), LEAK_PROBES))
        if not v.ok:
            rows.append((f, None, None, v.code, v.reason))
            deltas[f] = None
            continue

        r = measure(executor, store, f, baton, frozenset({f}), K)

        # G4 — missing data is never zero
        v = V.check_condition(r["errors"], K)
        if not v.ok:
            rows.append((f, None, None, v.code, v.reason))
            deltas[f] = None
            continue

        delta = (r["passes"] - base["passes"]) / K
        imp = V.importance(r["passes"], base["passes"], floor["passes"])
        deltas[f] = delta
        fails = {fl: n for fl, n in r["booked"].items() if fl != CORRECT_FLIGHT_ID}
        rows.append((f, delta, imp, verdict(imp), fails or "-"))

    # --- report ---
    print(f"=== cliff report ===  floor {floor['passes']}/{K}  baseline {base['passes']}/{K}"
          f"  ({time.perf_counter() - t0:.0f}s)")
    print(f"{'field':<18} {'Δ':>6} {'importance':>11}  {'verdict':<11} what B booked instead")
    for f, delta, imp, v, detail in rows:
        d = f"{delta:+.2f}" if delta is not None else "  -  "
        i = f"{imp:.2f}" if imp is not None else "  -  "
        print(f"{f:<18} {d:>6} {i:>11}  {v:<11} {detail}")

    # --- G5: controls ---
    print()
    v = V.check_controls(deltas, CONTROL_FIELDS)
    print(v.reason if not v.ok else "control fields flat at 0.00 — instrument behaving")
