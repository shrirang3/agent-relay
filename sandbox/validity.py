"""Validity gates. Every one of these exists because a run lied to us.

An ablation sweep is easy; knowing a delta means something is not. These gates are the
difference, and they are deliberately pure — numbers in, verdicts out, no I/O — so the
trustworthiness logic can be tested without spending a single token.
"""

from dataclasses import dataclass

OK = "OK"
UNTESTABLE = "UNTESTABLE"   # this field cannot be measured; NEVER report it as 0.00
INVALID = "INVALID"         # data is missing, not zero
ABORT = "ABORT"             # the whole sweep is meaningless, stop before spending calls


@dataclass(frozen=True)
class Verdict:
    code: str
    reason: str = ""

    @property
    def ok(self) -> bool:
        return self.code == OK


def check_floor(floor_passes: int, k: int) -> Verdict:
    """Empty baton. The receiver MUST fail without the note.

    The lower control. A sweep once ran 339s and returned 0.00 for all seven fields
    with a perfectly clean instrument — the receiver could solve the task from its own
    priors and the environment's labels, so no field could possibly register. Nothing
    in the harness could see that; this gate can, for the price of k trials.
    """
    if floor_passes >= k:
        return Verdict(ABORT, f"floor is {floor_passes}/{k} — the receiver succeeds with NO baton, "
                              f"so no field can matter. The task carries no signal.")
    return Verdict(OK)


def check_baseline(passes: int, errors: int, k: int) -> Verdict:
    """Full baton. The receiver MUST succeed every time.

    0/k has no success to lose, so every delta comes out >= 0 — once produced
    "removing the budget helps, +1.00". Anything below k/k means the receiver sits on
    its decision boundary, where it flips on any change to the prompt bytes; that run
    moved both planted control fields by exactly one trial.
    """
    if errors:
        return Verdict(ABORT, f"baseline had {errors}/{k} errored calls — fix rate limits first")
    if passes == 0:
        return Verdict(ABORT, f"baseline is 0/{k} — the receiver fails WITH the complete baton")
    if passes < k:
        return Verdict(ABORT, f"baseline is {passes}/{k} — receiver is at its decision boundary, "
                              f"ablations would measure prompt perturbation, not information")
    return Verdict(OK)


def check_field(value, leaks: list[str]) -> Verdict:
    """Can this field be ablated at all? Runs before any API call — both checks are free."""
    if not isinstance(value, bool) and not value:
        return Verdict(UNTESTABLE, "field is empty — removing it removes nothing")
    if leaks:
        return Verdict(UNTESTABLE, leaks[0])
    return Verdict(OK)


def check_condition(errors: int, k: int) -> Verdict:
    """A failed HTTP call is MISSING DATA, not a failed handoff.

    Scoring 429s as failures once reported `small_talk` — a planted control — as
    CRITICAL at -1.00. The control group caught a false positive manufactured by the
    scorer itself.
    """
    if errors:
        return Verdict(INVALID, f"{errors}/{k} calls errored — no data")
    return Verdict(OK)


def check_controls(deltas: dict[str, float | None], control_fields: tuple[str, ...]) -> Verdict:
    """Planted noise must read exactly 0.00. The upper control.

    Catches an over-sensitive instrument: if a field that cannot matter appears to
    matter, no other row in the table is trustworthy either.
    """
    moved = {f: d for f, d in deltas.items()
             if f in control_fields and d is not None and d != 0.0}
    if moved:
        return Verdict(INVALID, f"control fields moved: {moved} — do not trust this ranking")
    return Verdict(OK)


def importance(ablated_passes: int, baseline_passes: int, floor_passes: int) -> float | None:
    """Fraction of the baton's TOTAL contribution that this field carries.

    A raw delta is in units of "trials out of k", which cannot be compared across tasks
    or models. Dividing by the signal the baton actually provides (baseline - floor) can:
        1.0  -> this field alone accounts for everything the note contributes
        0.0  -> removing it changes nothing
        >1.0 -> the receiver does WORSE without this field than with no note at all,
                which means the remaining fields actively mislead it. Left unclamped
                because that is a finding, not an artefact.
    """
    signal = baseline_passes - floor_passes
    if signal <= 0:
        return None  # no signal to apportion; check_floor should already have aborted
    return (baseline_passes - ablated_passes) / signal


if __name__ == "__main__":
    # Zero API calls. This is the point of keeping the layer pure.
    assert check_floor(3, 3).code == ABORT
    assert check_floor(0, 3).ok
    assert check_baseline(3, 0, 3).ok
    assert check_baseline(2, 0, 3).code == ABORT
    assert check_baseline(0, 0, 3).code == ABORT
    assert check_baseline(3, 1, 3).code == ABORT
    assert check_field([], []).code == UNTESTABLE
    assert check_field(False, []).ok            # a bool is testable; its probes do the work
    assert check_field(500, ["leaked"]).code == UNTESTABLE
    assert check_condition(0, 3).ok
    assert check_condition(2, 3).code == INVALID
    assert check_controls({"user_mood": 0.0, "goal": -1.0}, ("user_mood",)).ok
    assert check_controls({"user_mood": -0.33}, ("user_mood",)).code == INVALID
    assert importance(0, 3, 0) == 1.0           # field carries the whole signal
    assert importance(3, 3, 0) == 0.0           # field carries none of it
    assert importance(3, 3, 3) is None          # no signal at all
    print("all validity gates pass")