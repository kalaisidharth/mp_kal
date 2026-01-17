
from typing import TypedDict
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
import os

from langchain_core.messages import HumanMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langgraph.prebuilt import create_react_agent   # UPDATED import

load_dotenv()

# LLM
llmg = ChatHuggingFace(
    llm=HuggingFaceEndpoint(repo_id='openai/gpt-oss-120b')
)

# ------------------- STATEGRAPH -------------------

class MyState(TypedDict):
    count: int

def increment(st: MyState) -> MyState:
    return {"count": st["count"] + 1}

graph = StateGraph(MyState)
graph.add_node("increment", increment)
graph.set_entry_point("increment")
graph.add_edge("increment", END)

app = graph.compile()
print(app.invoke({"count": 8}))

# ------------------- REACT AGENT -------------------

def add(a: int, b: int):
    """Adds two numbers."""
    return a + b

def multiply(a: int, b: int):
    """Multiplies two numbers."""
    return a * b

tools = [add, multiply]

react_agent = create_react_agent(model=llmg, tools=tools)

result_auto = react_agent.invoke({
    "messages": [HumanMessage(content="What is 12 + 30? Use tool.")]
})

print(result_auto)
