import os
from typing import Annotated, Sequence, TypedDict  #
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.messages import BaseMessage  # baseClass for all messagetypes in langGraph
from langchain_core.messages import ToolMessage  # Passes data back to LLM after it calls a tool
from langchain_core.messages import SystemMessage # Message for providing instructions to the LLM
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI 
from google.genai.types import AutomaticFunctionCallingConfig
from langgraph.graph import StateGraph, START, END  
from langgraph.graph.message import add_messages   # a reducer function.. helps merge data in the current state..
from langgraph.prebuilt import ToolNode

from dotenv import load_dotenv 

load_dotenv()  

class AgentState(TypedDict): 
    messages: Annotated[Sequence[BaseMessage], add_messages]

@tool 
def add(a:int, b:int): 
    """ this is an addition function """
    return a + b



tools = [add]

#setting up the agent/llm 
model = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key = os.getenv("GOOGLE_API_KEY")
).bind(
    automatic_function_calling=AutomaticFunctionCallingConfig(disable=True) # avoid the init messages...
)

llm = model.bind_tools(tools)   # bind the tools to the llm model. 

def model_call(state:AgentState) -> AgentState: 
    system_prompt = SystemMessage(content="You are my AI Assistant, answer my calls..")

    res = model.invoke([system_prompt] + state["messages"])   #system message + human messages. 

    return {"messages": [res.content[0]["text"]]}   # updates the state here... since the reducer message handles everything.. 


# define the conditional edge... 

def shld_continue(state:AgentState): 
    messages = state["messages"] 
    last_message = messages[-1]  #get the last message   

    if not last_message.tool_calls: 
        return "end" 
    else: 
        return "continue"


graph = StateGraph(AgentState)
graph.add_node("our_agent", model_call) 
graph.set_entry_point("our_agent")

tool_node = ToolNode(tools=tools)
graph.add_node("tools", tool_node)

graph.add_conditional_edges(
    "our_agent", 
    shld_continue, 
    {
        "continue": "tools", 
        "end": END,
    }
)

graph.add_edge("tools", "our_agent")


app = graph.compile() 




# helper function 

def print_stream(stream): 
    for s in stream: 
        message = s["messages"][-1]
        if isinstance(message, tuple): 
            print(message)
        else: 
            message.pretty_print()



inputs = {"messages": [("user", "Add 4 + 52")]}

print_stream(app.stream(inputs, stream_mode="values"))
