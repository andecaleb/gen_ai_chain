import os
from typing import TypedDict, List, Union
from langchain_core.messages import HumanMessage, AIMessage

from langchain_google_genai import (ChatGoogleGenerativeAI,GoogleGenerativeAIEmbeddings )
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from google.genai.types import AutomaticFunctionCallingConfig
from langgraph.graph import StateGraph, START, END  

from dotenv import load_dotenv 

load_dotenv() 

# 1. INITIALIZE LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY")
).bind(
    automatic_function_calling=AutomaticFunctionCallingConfig(
        disable=True
    )
)


# 2. LOAD DOCUMENT
with open("docs/ndr_doc.txt", "r", encoding="utf-8") as file:
    document = file.read()


# 3. SPLIT DOCUMENT INTO CHUNKS
text_splitter = RecursiveCharacterTextSplitter( chunk_size=500, chunk_overlap=50)

chunks = text_splitter.split_text(document)
print(f"Document split into {len(chunks)} chunks")


# 4. CREATE EMBEDDINGS
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001", google_api_key=os.getenv("GOOGLE_API_KEY"))


# 5. CREATE VECTOR DATABASE
vectorstore = FAISS.from_texts( chunks,embedding=embeddings)

# 6. CREATE RETRIEVER
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

# 7. LANGGRAPH STATE
class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]

# 8. PROCESS NODE
def process(state: AgentState) -> AgentState:

   
    question = state["messages"][-1].content    # Get the latest user question
    documents = retriever.invoke(question)      # RETRIEVE RELEVANT DOCUMENT CHUNKS 
    context = "\n\n".join(document.page_content for document in documents) # Combine retrieved chunks

    # CREATE RAG PROMPT
    prompt = f"""
    You are an AI assistant for the Nigerian National Data Repository (NDR).

    Your role is to help users understand and retrieve information from the NDR
    knowledge base, including datasets, data sources, government institutions,
    data owners, metadata, data standards, policies, indicators, statistics,
    data access procedures, and other information contained in the repository.

    Use ONLY the information provided in the context below to answer the user's
    question.

    Rules:
    1. Give accurate, clear, and concise answers based strictly on the context.
    2. Do not invent, assume, or fabricate information that is not present in
    the context.
    3. If the requested information cannot be found in the context, respond:
    "I don't have enough information in the NDR documents to answer that question."
    4. When possible, identify the relevant dataset, institution, data source,
    indicator, policy, or metadata described in the context.
    5. If the context contains multiple pieces of relevant information, combine
    them into a clear answer.
    6. Do not confuse similar datasets, institutions, indicators, or data sources.
    7. If the context provides dates, figures, definitions, or statistics,
    preserve them accurately.
    8. Do not use your general knowledge to fill gaps in the NDR documentation.
    9. If the user's question is ambiguous and the context does not provide
    enough information to determine the intended meaning, say so rather than
    making an assumption.
    10. Base every factual claim in your response on the supplied NDR context.

    NDR CONTEXT:
    ----------------
    {context}
    ----------------

    USER QUESTION:
    {question}

    Provide the most relevant answer using only the NDR context.
    """
    response = llm.invoke(prompt)   # SEND TO GEMINI
    answer = response.content       # Gemini's response content

   
    if isinstance(answer, list):    # Sometimes Gemini returns a list of content blocks
        answer = "".join(
            item.get("text", "")
            for item in answer
            if isinstance(item, dict)
        )


    # Add AI response to conversation
    state["messages"].append(
        AIMessage(content=answer)
    )


    print(f"\nAI: {answer}")

    return state


# 9. BUILD LANGGRAPH
graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)

agent = graph.compile()


# 10. CONVERSATION LOOP
conversation_history = []

user_input = input("Enter Message: ")

while user_input != "exit":

    conversation_history.append(HumanMessage(content=user_input))
    res = agent.invoke({ "messages": conversation_history})
    conversation_history = res["messages"]
    user_input = input("\nEnter Message: ")


# 11. SAVE CONVERSATION
with open("chatlogs.txt", "a", encoding="utf-8") as file:
    file.write("\nYour conversation log:\n")
    for message in conversation_history:

        if isinstance(message, HumanMessage):
            file.write(f"You: {message.content}\n")

        elif isinstance(message, AIMessage):
            file.write(f"AI: {message.content}\n")

    file.write("End of conversation\n")


print("Conversation saved to chatlogs.txt")