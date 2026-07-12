from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain.agents import create_agent

load_dotenv()
model = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

FLIGHTS = [
    {"id": "AI101", "price": 420, "red_eye": False},   # valid
    {"id": "AI202", "price": 380, "red_eye": True},    # cheaper BUT red-eye
    {"id": "AI303", "price": 610, "red_eye": False},   # over budget
]
BOOKED = []  # we'll record what B books, to check it

@tool
def list_flights() -> list:
    """List all available flights."""
    return FLIGHTS

@tool
def book_flight(flight_id: str) -> str:
    """Book a flight by its ID."""
    for flight in FLIGHTS:
        if flight["id"] == flight_id:
            BOOKED.append(flight)
            return f"Flight {flight_id} booked successfully."
    return f"Flight {flight_id} not found."


executor = create_agent(
    model,
    tools=[list_flights, book_flight],
    system_prompt=(
        "You are a flight booker. ALWAYS call list_flights FIRST to get real flight ids. "
        "Never invent a flight id. Pick the flight satisfying ALL constraints, then call "
        "book_flight with its real id."
    ),
)

# result = executor.invoke({"messages": [(
#     "user",
#     "Book me a flight. Constraints: budget under $500, and NO red-eye flights.",
# )]})

# print(result)


# --- Agent A: Planner (no tools, just understands + plans) ---
planner = create_agent(
    model,
    tools=[],
    system_prompt=(
        "You are a travel planner. Chat with the user, understand their needs, "
        "and produce a short plan of what to book. You do NOT book anything yourself."
    ),
)

# a messy, multi-turn user request: constraints buried in noise + a mid-way UPDATE
user_convo = [
    ("user", "Hey! Planning a quick weekend trip, pretty pumped. Budget's around $600 I reckon."),
    ("user", "Ugh, weather's been miserable here all week, unrelated but venting."),
    ("user", "Oh — important: I absolutely cannot do red-eye flights, they wreck me."),
    ("user", "Actually scratch the $600 — keep it UNDER $500, money's tight this month."),
    ("user", "That's everything, sort me out!"),
]

a_result = planner.invoke({"messages": user_convo})

# print("=== A (Planner) full message list ===")
# for m in a_result["messages"]:
#     m.pretty_print()

baton_full_dump = a_result["messages"]          # everything A saw + said = the "note"

# BOOKED.clear()                                   # reset so we measure THIS handoff only
# handoff_input = {"messages": baton_full_dump + [("user", "Book the flight per the conversation above.")]}
# b_result = executor.invoke(handoff_input)

# print("\n=== B (Executor) after FULL-DUMP handoff ===")
# for m in b_result["messages"]:
#     m.pretty_print()

# print("\nBOOKED (full-dump):", BOOKED)           # correct = ['AI101']

# # how expensive was the dump? read the REAL input tokens B ingested
# for m in b_result["messages"]:
#     um = getattr(m, "usage_metadata", None)
#     if um:
#         print("B input tokens on a call:", um["input_tokens"])

from pydantic import BaseModel

class Baton(BaseModel):
    """The note passed A → B. Small, typed, only what B needs."""
    goal: str
    constraints: list[str]

extractor = model.with_structured_output(Baton)
baton = extractor.invoke(
    [("system",
      "Extract the user's booking request into the note. "
      "If the user changed their mind, use the LATEST value. Ignore chit-chat.")]
    + user_convo
)

print("=== Baton A produced from the convo ===")
print(baton.model_dump_json(indent=2))

BOOKED.clear()
b_struct = executor.invoke({"messages": [(
    "user",
    f"Book a flight for this request:\n{baton.model_dump_json(indent=2)}",
)]})

print("\n=== B after STRUCTURED baton ===")
for m in b_struct["messages"]:
    m.pretty_print()

print("\nBOOKED (structured):", BOOKED)
for m in b_struct["messages"]:
    um = getattr(m, "usage_metadata", None)
    if um:
        print("B input tokens on a call:", um["input_tokens"])