import os
from langchain_google_genai import ChatGoogleGenerativeAI 
from google.genai.types import AutomaticFunctionCallingConfig
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv() 

model = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key = os.getenv("GOOGLE_API_KEY")
).bind(
    automatic_function_calling=AutomaticFunctionCallingConfig(disable=True) #
)

chat_history = [
    SystemMessage(content="You are a helful AI Assistant")
]

while True:  
    user_input=input("You: ")
    chat_history.append(HumanMessage(content=user_input))
    if user_input.lower() == "exit": 
        break; 
    res = model.invoke(chat_history)            # invoke the gemini model on the chat history...... 
    ctxt = res.content[0]["text"]               # extract the text property of the firsst content item. 
    chat_history.append(AIMessage(content=ctxt))# append it to the chat history... as an AI message.. 
    print("Bot:", ctxt)

print("chat history", chat_history)