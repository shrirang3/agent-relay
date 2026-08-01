import json
import pathlib
import time

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from agents import make_planner

from fixtures import make_model, user_convo


class Baton(BaseModel):
    """The note passed A → B. Fixed schema — only which fields are PRESENT varies."""

    # --- expected load-bearing ---
    goal: str = Field(description="The single action B must perform, one line, starting with an imperative verb B can execute. Not a planning step. The ACTION ONLY — no constraints, no prices, no numbers, no budget. Write 'Book a flight', NEVER 'Book a flight under $500'.")
    budget_usd: int | None = Field(default=None, description="Hard maximum in USD. The LATEST value if the user changed it.")
    avoid_red_eye: bool | None = Field(default=None, description="True if the user refuses red-eye flights.")

    # --- unknown, this is what we're here to find out ---
    open_steps: list[str] = Field(default_factory=list, description="Concrete next actions for B.")
    superseded: list[str] = Field(default_factory=list, description="Values the user stated then ABANDONED, e.g. 'budget was 600'. Record only the old value — never the replacement that took its place.")

    # --- planted noise: CONTROL GROUP, must rank Δ0.0 ---
    user_mood: str = Field(default="", description="The user's emotional tone in one or two words. No task details, no constraints.")
    small_talk: str = Field(default="", description="Off-topic chatter, near-verbatim. Off-topic ONLY — nothing that states or hints at a constraint.")

    def to_prompt(self, drop: frozenset[str] = frozenset()) -> str:
        """Serialize for B, omitting `drop` fields entirely (not nulled)."""
        assert drop <= set(FIELDS), f"unknown field(s): {drop - set(FIELDS)}"
        return self.model_dump_json(exclude=set(drop), indent=None)


# every field name, for the cliff sweep to iterate
FIELDS = tuple(Baton.model_fields)


EXTRACT_SYSTEM = (
    "You compress a working agent's transcript into a handoff note for the agent that takes "
    "over. That agent will NEVER see this transcript — only your note. Anything missing from "
    "the note is lost to it.\n"
    "`open_steps` comes from the working agent's PLAN, not from the user: the concrete actions "
    "it decided on but did not carry out. If it planned nothing actionable, leave it empty.\n"
    "If the user changed their mind, put the LATEST value in the field and record the "
    "abandoned one in `superseded` as a short phrase (e.g. 'budget was 600').\n"
    "Fill EVERY field the transcript supports, including `user_mood` and `small_talk` — "
    "capture off-topic chatter near-verbatim instead of discarding it.\n"
    "Never invent. If the transcript never stated something, leave that field at its default.\n"
    "Each fact belongs to EXACTLY ONE field. Never restate a constraint inside `goal`, "
    "`superseded`, `user_mood`, or `small_talk` — a fact repeated in two fields cannot be removed "
    "from the note, and removing facts is the whole purpose of this schema."
)

# Domain vocabulary: facts that leak as WORDING rather than as a value.
# Passed in to find_leaks, never read as a global — keeps find_leaks generic.
LEAK_PROBES = {
    "avoid_red_eye": ("red-eye", "red eye", "redeye", "overnight"),
}


def find_leaks(
    b: "Baton",
    drop: frozenset[str],
    probes: dict[str, tuple[str, ...]] | None = None,
) -> list[str]:
    """After dropping `drop`, which dropped facts still appear in the note anyway?

    A leak makes an ablation a no-op, which reports as Δ0.0 — indistinguishable from
    "this field doesn't matter". Silent false negatives are the failure mode this
    guards against.

    Knows nothing about this schema; `probes` supplies the per-field domain vocabulary.
    Returns leak descriptions — an empty list means the ablation is clean.
    """
    probes = probes or {}
    surviving = b.to_prompt(drop=drop).lower()
    leaks: list[str] = []

    for field in sorted(drop):
        value = getattr(b, field)

        # layer 1 — the value itself survived elsewhere in the note
        items = value if isinstance(value, (list, tuple, set)) else [value]
        for item in items:
            if isinstance(item, bool):
                continue  # str(True) == "true" matches every other bool in the JSON
            if not item:
                continue  # nothing to leak; also "" is a substring of everything
            if str(item).lower() in surviving:
                leaks.append(f"{field}: value {item!r} still in note")

        # layer 2 — the concept survived as wording
        for probe in probes.get(field, ()):
            if probe in surviving:
                leaks.append(f"{field}: probe {probe!r} still in note")

    return leaks

def all_leaks(b: "Baton", probes=None) -> list[str]:
    """Every leak across every single-field ablation. Empty = baton is sweep-ready."""
    found = []
    for f in FIELDS:
        found.extend(find_leaks(b, frozenset({f}), probes))
    return found


def extract_baton(model, transcript) -> Baton:
    """Compress A's working state into the note.

    `transcript` is A's full message list — the user turns AND A's plan. Extracting from
    A's output rather than the raw user chat is what makes this a real handoff: the note
    now depends on work A actually did, and `open_steps` has a source (users state goals,
    planners state next actions).
    """
    extractor = model.with_structured_output(Baton)
    return extractor.invoke([("system", EXTRACT_SYSTEM)] + list(transcript))


def extract_clean_baton(model, transcript, probes=None, max_attempts=2):
    """Extract a baton, then verify no single-field ablation is a no-op.

    Retries with a corrective message naming the offending fields. Returns
    (baton, attempts, remaining_leaks) — a non-empty third value means those
    fields are UNTESTABLE, not unimportant, and the sweep must say so.
    """
    extractor = model.with_structured_output(Baton)
    messages = [("system", EXTRACT_SYSTEM)] + list(transcript)

    for attempt in range(1, max_attempts + 1):
        b = extractor.invoke(messages)
        leaks = all_leaks(b, probes)
        if not leaks:
            return b, attempt, []
        # Append the correction. At temperature=0 a bare retry returns the
        # identical baton — changing the input is the only thing that changes
        # the output.
        messages = messages + [(
            "system",
            "Your previous note repeated the same fact in more than one field:\n"
            + "\n".join(f"- {x}" for x in leaks)
            + "\nRewrite it so each fact appears in EXACTLY ONE field. Strip the "
              "duplicated value out of every field except the one that owns it.",
        )]

    return b, max_attempts, leaks


# A's transcript doesn't change while we iterate on the baton, so don't pay for it
# every run. Same idea as Redis role 2 in the plan — replay stored state instead of
# re-running A. Delete the file to force a fresh planner run.
TRANSCRIPT_CACHE = pathlib.Path(__file__).parent / ".transcript.json"


def get_transcript(planner, convo, cache: pathlib.Path = TRANSCRIPT_CACHE):
    """A's full message list as (role, content) pairs, cached on disk."""
    if cache.exists():
        return [tuple(pair) for pair in json.loads(cache.read_text())]
    msgs = planner.invoke({"messages": convo})["messages"]
    pairs = [(m.type, str(m.content)) for m in msgs]
    cache.write_text(json.dumps(pairs))
    return pairs


if __name__ == "__main__":
    load_dotenv()  # must precede make_model — ChatGroq reads GROQ_API_KEY at construction
    model = make_model()

    # A works the task first; the baton compresses what A ended up with,
    # not what the user typed.
    planner = make_planner(model)
    was_cached = TRANSCRIPT_CACHE.exists()
    t = time.perf_counter()
    transcript = get_transcript(planner, user_convo)
    print(f"[A] {time.perf_counter() - t:.1f}s  ({'cached' if was_cached else 'fresh'})")

    t = time.perf_counter()
    b, attempts, leaks = extract_clean_baton(model, transcript, LEAK_PROBES)
    print(f"[extract] {time.perf_counter() - t:.1f}s  attempts={attempts}  unresolved={leaks or 'none'}\n")

    print(b.to_prompt())
    print(b.to_prompt(drop=frozenset({"budget_usd"})))  # prove the key vanishes

    print("\n=== leak self-test: drop one field at a time ===")
    for f in FIELDS:
        found = find_leaks(b, frozenset({f}), LEAK_PROBES)
        print(f"{f:<14} {'clean' if not found else found}")
