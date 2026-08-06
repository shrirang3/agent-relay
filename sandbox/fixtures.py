"""Shared fixture: the flight-booking trap that A hands off to B.

One source of truth — imported by agent2.py, baton.py, and later measure.py / cliff.py.
Importing this module makes NO network calls.
"""

from langchain_core.tools import tool

# pin the model in one place; never let ChatGroq default it
MODEL = "llama-3.3-70b-versatile"

# FLIGHTS = [
#     {"id": "AI101", "price": 420, "red_eye": False},  # valid — the only correct answer
#     {"id": "AI202", "price": 380, "red_eye": True},   # cheaper BUT red-eye -> booked when "no red-eye" is lost
#     {"id": "AI303", "price": 610, "red_eye": False},  # over budget -> booked when the stale $600 survives
# ]

FLIGHTS = [
    # Constraints are deliberately ARBITRARY. The previous fixture used `red_eye`, and
    # every field scored 0.00 because B avoids red-eyes on its own — the environment's
    # own label carried the constraint, so removing it from the note changed nothing.
    # A model has no prior about terminal numbers or refundability, so those facts can
    # only reach B through the baton.
    #
    # AI101 is also the MOST EXPENSIVE, so a price-minimising receiver cannot reach it
    # by default. Each decoy is cheaper and violates exactly one constraint, which makes
    # each constraint independently decisive:
    #   drop required_terminal -> cheapest refundable is AI202  -> wrong
    #   drop needs_refundable  -> cheapest terminal-2 is AI303  -> wrong
    #   drop both / neither    -> AI202 (cheapest overall)      -> wrong
    {"id": "AI101", "price": 460, "terminal": 2, "refundable": True},   # CORRECT
    {"id": "AI202", "price": 380, "terminal": 1, "refundable": True},   # cheaper, wrong terminal
    {"id": "AI303", "price": 400, "terminal": 2, "refundable": False},  # cheaper, not refundable
]

HOTELS = [
    {"id": "HT9", "price": 120, "nights": 2},
]

CORRECT_FLIGHT_ID = "AI101"

BOOKED = []  # ground-truth probe: B's tool call mutates this

def make_model(**kw):
    """One place to build the model. timeout so a stalled call fails instead of
    blocking a sweep; retries low by default because backoff hides real latency.

    setdefault, not literals — passing any of these explicitly (cliff.py raises
    max_retries for sweeps) would otherwise collide with the hardcoded value.
    """
    from langchain_groq import ChatGroq

    kw.setdefault("model", MODEL)
    kw.setdefault("temperature", 0)
    kw.setdefault("timeout", 90)
    kw.setdefault("max_retries", 2)
    return ChatGroq(**kw)

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

@tool
def list_hotels() -> list:
    """List all available hotels."""
    return HOTELS


@tool
def book_hotel(hotel_id: str) -> str:
    """Book a hotel by its ID."""
    for h in HOTELS:
        if h["id"] == hotel_id:
            BOOKED.append(h)
            return f"Hotel {hotel_id} booked successfully."
    return f"Hotel {hotel_id} not found."


# a messy, multi-turn user request: constraints buried in noise + a mid-way UPDATE
user_convo = [
    ("user", "Hey! Planning a quick weekend trip, pretty pumped. Budget's around $600 I reckon."),
    ("user", "Ugh, weather's been miserable here all week, unrelated but venting."),
    ("user", "Oh — important: I absolutely cannot do red-eye flights, they wreck me."),
    ("user", "Actually scratch the $600 — keep it UNDER $500, money's tight this month."),
    ("user", "That's everything, sort me out!"),
]
