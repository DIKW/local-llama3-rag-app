# from langchain_community.llms import Ollama
import logging
from langchain_community.chat_models import ChatOllama

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings

from langchain.callbacks.base import BaseCallbackHandler


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
    memory = ConversationBufferMemory(memory_key="chat_history", output_key="answer" ,return_messages=True)

    return memory


def get_retriever(context):
    """
    Returns the retriever.

    :return: The retriever.
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

        retriever = faiss_vector_db.as_retriever(search_kwargs={"k": 10})
        logging.debug(f"Retriever: {retriever}")
    except Exception as e:
        logging.error(e)

    return retriever


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

    # setup the RAG chain
    from langchain.chains import ConversationalRetrievalChain

    retriever = get_retriever(context)

    memory = get_memory()

    prompt = get_custom_prompt()

    qa_client = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt},
        verbose=True,
        # return_source_documents=True,
        # output_key="answer",
    )

    return qa_client
