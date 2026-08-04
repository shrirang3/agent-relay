from langchain.agents import create_agent
from fixtures import list_flights, book_flight, list_hotels, book_hotel


PLANNER_SYSTEM = (
    "You are a travel planner. Talk with the user, work out what they need, and produce a "
    "short plan — no more than 100 words. End with a line 'Next actions:' followed by 2-4 "
    "concrete steps for whoever executes the booking.\n"
    "You do NOT book anything yourself, and you have no access to any listings — so never "
    "name a specific flight, hotel, destination, or price. Inventing them would put fiction "
    "into the handoff for the next agent to act on."
)

# Tool hygiene + fidelity to the note, and the difference between those two is the whole
# design. Deliberately absent:
#   - task identity ("you are a flight booker") -> would survive ablation of `goal`
#   - "satisfy ALL constraints" -> B would honour constraints the baton never supplied,
#     deflating every constraint field's delta
# But silence went too far: with no binding instruction at all, B treated the note's
# constraints as preferences and took the cheaper, faster, nonstop red-eye — the baseline
# failed 0/3 with the COMPLETE baton, which makes every field unmeasurable.
# "Every field in the note is binding" is the resolution. It has force only for fields
# that are present, so ablating one genuinely removes B's reason to comply.
EXECUTOR_SYSTEM = (
    "You execute a handoff note written by another agent. Every field in the note is a hard "
    "requirement, not a preference — never trade one off against convenience, price, or speed. "
    "If the note does not mention something, you have no information about it. "
    "Before using any id, call the matching list tool to get the real ones — never invent "
    "or guess an id."
)


def make_planner(model):
    """Agent A — works the task, holds no tools, hands off."""
    return create_agent(
        model,
        tools=[],
        system_prompt=PLANNER_SYSTEM,
    )

def make_executor(model):
    return create_agent(
        model,
        tools=[list_flights, book_flight, list_hotels, book_hotel],
        system_prompt=EXECUTOR_SYSTEM,
    )
    