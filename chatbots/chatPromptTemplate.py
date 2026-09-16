from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI 
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

#define the model prompt.. 
chat_template = ChatPromptTemplate([
    ('system', "You are a helful {domain} expert. "), 
    ('human', "Explain in simple terms, the concept of {topics} ")
])

prompt = chat_template.invoke({
    'domain': "quantum physics", 
    'topics': 'wormhole'
})

print(prompt)

#invoke the model here... 
# result = model.invoke(prompt)