from dotenv import load_dotenv
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

from fixtures import MODEL, user_convo


class Baton(BaseModel):
    """The note passed A → B. Fixed schema — only which fields are PRESENT varies."""

    # --- expected load-bearing ---
    goal: str = Field(description="The single action B must perform, one line, starting with an imperative verb B can execute (e.g. 'Book ...'). Not a planning step. The ACTION ONLY — no constraints, no prices, no numbers. Those live in their own fields.")
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
    "You extract a handoff note for a second agent. That agent will NEVER see this "
    "conversation — only your note. Anything missing from the note is lost to it.\n"
    "If the user changed their mind, put the LATEST value in the field and record the "
    "abandoned one in `superseded` as a short phrase (e.g. 'budget was 600, now 500').\n"
    "Fill EVERY field the conversation supports, including `user_mood` and `small_talk` — "
    "capture off-topic chatter near-verbatim instead of discarding it.\n"
    "Never invent. If the conversation never stated something, leave that field at its default.\n"
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


def extract_baton(model, convo: list[tuple[str, str]]) -> Baton:
    """A produces the baton from the raw conversation."""
    extractor = model.with_structured_output(Baton)
    return extractor.invoke([("system", EXTRACT_SYSTEM)] + convo)


if __name__ == "__main__":
    load_dotenv()
    model = ChatGroq(model=MODEL, temperature=0)
    b = extract_baton(model, user_convo)
    print(b.to_prompt())
    print(b.to_prompt(drop=frozenset({"budget_usd"})))  # prove the key vanishes

    print("\n=== leak self-test: drop one field at a time ===")
    for f in FIELDS:
        found = find_leaks(b, frozenset({f}), LEAK_PROBES)
        print(f"{f:<14} {'clean' if not found else found}")
