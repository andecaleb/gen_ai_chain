import os
from typing import TypedDict, List, Union
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI 
from google.genai.types import AutomaticFunctionCallingConfig
from langgraph.graph import StateGraph, START, END  

from dotenv import load_dotenv 

load_dotenv() 

# initialize the LLM here... 

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key = os.getenv("GOOGLE_API_KEY")
).bind(
    automatic_function_calling=AutomaticFunctionCallingConfig(disable=True) # avoid the init messages...
)


# Union - ensures either variable type is captured.... 

class AgentState(TypedDict): 
    messages: List[Union[HumanMessage, AIMessage]]  #allows to store human or ai messages in the messages property.. 
    

def process(state:AgentState) -> AgentState: 
    """ This node will solve the request you input """
    response = llm.invoke(state["messages"])                                  # messages = AI or Human message
    state["messages"].append(AIMessage(content=response.content[0]["text"]))  # append the ai message output... 
    print(f"\nAI: {response.content[0]["text"]}")                             # for the terminal
    return state



graph = StateGraph(AgentState)

graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)

agent = graph.compile()


conversation_history  = []  #initialize the conversation history... 

# to keep receiving a message until the user types exit... use a while loop. 
user_input = input("Enter Message: ")
while user_input != "exit": 
    conversation_history.append(HumanMessage(content=user_input))
    res = agent.invoke({"messages": conversation_history})
 
    conversation_history = res["messages"]    # replace the converation history completely.... 

    user_input = input("Enter Message: ")


    # store the data in a text file.. so when you exit it still remembers... 
 

#with open("chatlogs.txt", "w") as file: 
with open("chatlogs.txt", "a", encoding="utf-8") as file:
    file.write("Your conversation log : \n")
    for message in conversation_history: 
        if isinstance(message, HumanMessage): 
            file.write(f"You: {message.content}\n")
        elif isinstance(message, AIMessage): 
            file.write(f"AI: {message.content}\n")
    file.write("End of conversation")

print("Conversation saved to chatlogs.txt")