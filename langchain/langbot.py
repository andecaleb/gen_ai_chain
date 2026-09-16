import os
from typing import TypedDict, List 
from langchain_core.messages import HumanMessage 
from langchain_google_genai import ChatGoogleGenerativeAI 
from google.genai.types import AutomaticFunctionCallingConfig
from langgraph.graph import StateGraph, START, END 
from dotenv import load_dotenv 


load_dotenv() 

model = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key = os.getenv("GOOGLE_API_KEY")
).bind(
    automatic_function_calling=AutomaticFunctionCallingConfig(disable=True) # avoid the init messages...
)

class AgentState(TypedDict): 
    messages: List[HumanMessage]

def process(state:AgentState) -> AgentState: 
    response = model.invoke(state["messages"])
    print(f"\nAI: {response.content}")
    return state

graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)

agent = graph.compile()