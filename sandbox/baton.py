from dotenv import load_dotenv
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

from fixtures import MODEL, user_convo


class Baton(BaseModel):
    """The note passed A → B. Fixed schema — only which fields are PRESENT varies."""

    # --- expected load-bearing ---
    goal: str = Field(description="What B must accomplish, one line.")
    budget_usd: int | None = Field(default=None, description="Hard maximum in USD. The LATEST value if the user changed it.")
    avoid_red_eye: bool | None = Field(default=None, description="True if the user refuses red-eye flights.")

    # --- unknown, this is what we're here to find out ---
    open_steps: list[str] = Field(default_factory=list, description="Concrete next actions for B.")
    superseded: list[str] = Field(default_factory=list, description="Constraints the user stated then changed, e.g. 'budget was 600, now 500'.")

    # --- planted noise: CONTROL GROUP, must rank Δ0.0 ---
    user_mood: str = Field(default="", description="The user's emotional tone.")
    small_talk: str = Field(default="", description="Off-topic chatter, near-verbatim.")

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
    "Never invent. If the conversation never stated something, leave that field at its default."
)


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
