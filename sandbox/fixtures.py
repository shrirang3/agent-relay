"""Shared fixture: the flight-booking trap that A hands off to B.

One source of truth — imported by agent2.py, baton.py, and later measure.py / cliff.py.
Importing this module makes NO network calls.
"""

from langchain_core.tools import tool

# pin the model in one place; never let ChatGroq default it
MODEL = "llama-3.3-70b-versatile"

FLIGHTS = [
    {"id": "AI101", "price": 420, "red_eye": False},  # valid — the only correct answer
    {"id": "AI202", "price": 380, "red_eye": True},   # cheaper BUT red-eye -> booked when "no red-eye" is lost
    {"id": "AI303", "price": 610, "red_eye": False},  # over budget -> booked when the stale $600 survives
]

CORRECT_FLIGHT_ID = "AI101"

BOOKED = []  # ground-truth probe: B's tool call mutates this

def make_model(**kw):
    """One place to build the model. timeout so a stalled call fails instead of
    blocking a sweep; retries left low because backoff hides real latency."""
    from langchain_groq import ChatGroq
    return ChatGroq(model=MODEL, temperature=0, timeout=90, max_retries=2, **kw)

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


# a messy, multi-turn user request: constraints buried in noise + a mid-way UPDATE
user_convo = [
    ("user", "Hey! Planning a quick weekend trip, pretty pumped. Budget's around $600 I reckon."),
    ("user", "Ugh, weather's been miserable here all week, unrelated but venting."),
    ("user", "Oh — important: I absolutely cannot do red-eye flights, they wreck me."),
    ("user", "Actually scratch the $600 — keep it UNDER $500, money's tight this month."),
    ("user", "That's everything, sort me out!"),
]
