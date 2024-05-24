# from langchain_community.llms import Ollama
import os
import logging
from langchain_community.chat_models import ChatOllama

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings

from langchain.callbacks.base import BaseCallbackHandler

import streamlit as st


def get_custom_prompt():
    """
    Returns a custom prompt.

    TODO make it context aware

    :return: The custom prompt.
    """
    # setup the prompt
    from langchain.prompts import PromptTemplate

    # Build prompt
    template = """Gebruik de volgende stukjes context om de vraag aan het einde te beantwoorden. Als je het antwoord niet weet, zeg dan gewoon dat je het niet weet, probeer geen antwoord te verzinnen. 
    Gebruik maximaal drie zinnen. Houd het antwoord zo beknopt mogelijk. Zeg altijd "bedankt voor het vragen!" aan het einde van het antwoord. Antwoord in het Nederlands.
    {context}
    Vraag: {question}
    Waardevol antwoord:"""

    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

    return QA_CHAIN_PROMPT


def get_memory():
    """
    Returns the memory.

    :return: The memory.
    """
    # setup memory
    from langchain.memory import ConversationBufferMemory

    # explicitly store the chat history in the memory use output key to store the answer only
    memory = ConversationBufferMemory(
        memory_key="chat_history", output_key="answer", return_messages=True
    )

    return memory


def get_retriever(context, k=5):
    """
    Returns the retriever.

    :return: The retriever.
    """

    retriever = []
    # Laad de documenten uit de data directory in de vectorstore
    try:
        embeddings = OllamaEmbeddings(
            base_url="http://pcloud:11434", model="nomic-embed-text"
        )
        # path to index
        path = f"contexts/{context}"
        faiss_vector_db = FAISS.load_local(
            path, embeddings, allow_dangerous_deserialization=True
        )

        retriever = faiss_vector_db.as_retriever(search_kwargs={"k": k})
        logging.debug(f"Retriever: {retriever}")
    except Exception as e:
        logging.error(e)

    return retriever


def get_vector_db(context):
    """
    Returns the vector db from context.

    :return: The vector db.
    """

    # Laad de documenten uit de data directory in de vectorstore
    try:
        embeddings = OllamaEmbeddings(
            base_url="http://pcloud:11434", model="nomic-embed-text"
        )
        # path to index
        path = f"contexts/{context}"
        faiss_vector_db = FAISS.load_local(
            path, embeddings, allow_dangerous_deserialization=True
        )

        logging.debug(f"Get vector db from context: {context}")
    except Exception as e:
        logging.error(e)

    return faiss_vector_db


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs):
        self.text += token
        self.container.markdown(self.text)


def get_custom_rag_chain(context: str):
    """
    Returns a RAG chain based on the context.

    :param context: The context to use.

    Returns:
        The RAG chain.
    """
    # setup the chat model
    llm = ChatOllama(base_url="http://pcloud:11434", model="llama3", temperature=0.0)

    retriever = get_retriever(context)

    prompt = get_custom_prompt()

    # setup the RAG chain

    ############  DEPRECIATED
    # from langchain.chains import ConversationalRetrievalChain

    # memory = get_memory()

    # qa_client = ConversationalRetrievalChain.from_llm(
    #     llm,
    #     retriever=retriever,
    #     memory=memory,
    #     combine_docs_chain_kwargs={"prompt": prompt},
    #     verbose=True,
    #     # return_source_documents=True,
    #     # output_key="answer",
    # )

    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # simple chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # from langchain_core.runnables import RunnableParallel

    # rag_chain_from_docs = (
    #     RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
    #     | prompt
    #     | llm
    #     | StrOutputParser()
    # )

    # rag_chain_with_source = RunnableParallel(
    #     {"context": retriever, "question": RunnablePassthrough()}
    # ).assign(answer=rag_chain_from_docs)

    return rag_chain


def get_contexts() -> tuple:
    """
    Returns the contexts available in the system.

    :return: A tuple containing the contexts.
    """
    # we start with a default "free format" context plus we add  all contexts available in the context folder
    contexts = ["free format"]
    # get all sub directory names of the directotry contexts
    context_folders = os.listdir("./contexts")
    # add all sub directory names to the contexts list
    contexts.extend(context_folders)

    return tuple(contexts)


def set_new_context():
    """
    Clears the chat history.
    """
    # set new context to True
    st.session_state.new_context = True
    return
