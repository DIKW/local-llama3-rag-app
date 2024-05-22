import streamlit as st

# import ollama

# from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOllama

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings

# streamlit has logging available
# set streamlit logger
import logging

logging.basicConfig(level=logging.DEBUG)

# Laad de documenten uit de data directory in de vectorstore
try:
    embeddings = OllamaEmbeddings(
        base_url="http://pcloud:11434", model="nomic-embed-text"
    )
    faiss_vector_db = FAISS.load_local(
        "faiss_index", embeddings, allow_dangerous_deserialization=True
    )

    retriever = faiss_vector_db.as_retriever(search_kwargs={"k": 10})
    logging.debug(f"Retriever: {retriever}")
except Exception as e:
    logging.error(e)

# setup the chat model
llm = ChatOllama(base_url="http://pcloud:11434", model="llama3", temperature=0.0)

# setup memory
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# setup the prompt
from langchain.prompts import PromptTemplate

# Build prompt
template = """Gebruik de volgende stukjes context om de vraag aan het einde te beantwoorden. Als je het antwoord niet weet, zeg dan gewoon dat je het niet weet, probeer geen antwoord te verzinnen. 
Gebruik maximaal drie zinnen. Houd het antwoord zo beknopt mogelijk. Zeg altijd "bedankt voor het vragen!" aan het einde van het antwoord. Antwoord in het Nederlands.
{context}
Vraag: {question}
Waardevol antwoord:"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)


# setup the RAG chain
from langchain.chains import ConversationalRetrievalChain

# retriever=faiss_vector_db.as_retriever()

qa = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory)

st.title("Chat with your documents using Llama-3 and RAG")
st.caption(
    "This app allows you to chat with your documents using local Llama-3 and RAG"
)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Ask a question about the webpage
question = st.text_input("Stel een vraag aan de documenten")

# Chat with the documents
if question:
    result = qa({"question": question})
    st.write(result["answer"])

