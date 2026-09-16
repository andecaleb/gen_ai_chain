import os
from typing import TypedDict, List, Union
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI 
from google.genai.types import AutomaticFunctionCallingConfig
from langgraph.graph import StateGraph, START, END 
from IPython.display import Image, display  

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
    response = model.invoke(state["messages"])   # llms are invoked in functions thats how it is used... 
    print(f"\nAI: {response.content[0]["text"]}")
    return state

graph = StateGraph(AgentState)

graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)

agent = graph.compile()

display(Image(agent.get_graph().draw_mermaid_png()))  #render the image output.. here.. 


# this receives a message and stops..... 
# user_input = input("Enter: ")
# agent.invoke({"messages": [HumanMessage(content=user_input)]})

# to keep receiving a message until the user types exit... use a while loop. 
user_input = input("Enter Message: ")
while user_input != "exit": 
    agent.invoke({"messages": [HumanMessage(content=user_input)]})
    user_input = input("Enter Message: ")