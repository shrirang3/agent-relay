from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain.agents import create_agent

load_dotenv()

model=ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)

@tool
def get_word_length(word: str) -> int:
    """Get the length of a word."""
    return len(word)

agent = create_agent(model, tools=[get_word_length])

result=agent.invoke({"messages": [{"role": "user", "content": "How many letters are there in word: 'Olmo'?"}]})

print(result)