from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI 
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

chat_template = ChatPromptTemplate([
    'system', "you are a very helful customer support agent", 
    MessagesPlaceholder(variable_name="chat_history"), 
    ('human', '{query}')
])

#load chat history
chat_history = []
with open('chatbot_history.txt') as file: 
    chat_history.extend(file.readlines())

prompt = chat_template.invoke({
    'chat_history': chat_history, 
    'query': 'where is my refund?'
})

print(prompt)