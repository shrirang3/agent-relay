from langchain.agents import create_agent
from fixtures import list_flights, book_flight


PLANNER_SYSTEM = (
    "You are a travel planner. Talk with the user, work out what they need, and produce a "
    "short, concrete plan of what to book. You do NOT book anything yourself."
)

# Tool hygiene only. Deliberately absent:
#   - task identity ("you are a flight booker") -> would survive ablation of `goal`
#   - "satisfy ALL constraints" -> worse: B would honour constraints the baton never
#     mentioned, silently deflating every constraint field in the ranking
EXECUTOR_SYSTEM = (
    "You execute a handoff note written by another agent. Do exactly what the note says. "
    "Before using any id, call list_flights to get the real ones — never invent or guess an id."
)


def make_planner(model):
    """Agent A — works the task, holds no tools, hands off."""
    return create_agent(
        model,
        tools=[],
        system_prompt=PLANNER_SYSTEM,
    )

def make_executor(model):
    """Agent B — executes the baton, has tools, no planning."""
    return create_agent(
        model,
        tools=[list_flights, book_flight],
        system_prompt=EXECUTOR_SYSTEM,
    )
    