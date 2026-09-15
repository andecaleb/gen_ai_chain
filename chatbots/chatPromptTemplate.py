from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI 
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

chat_template = ChatPromptTemplate([
    ('system', "You are a helful {domain} expert. "), 
    ('human', "Explain in simple terms, the concept of {topics} ")
])

prompt = chat_template.invoke({
    'domain': "quantum physics", 
    'topic': 'wormhole'
})

print(prompt)