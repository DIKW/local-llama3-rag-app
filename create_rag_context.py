# Importeer de benodigde modules
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Laad de documenten uit de data directory waarvoor je een context wil maken
# loader = DirectoryLoader(path="./data", glob="./*.pdf", loader_cls=PyPDFLoader)
#loader = PyPDFLoader("./data/Rapport-PEFD-Blind-voor-mens-en-recht-26022024.pdf")
loader = PyPDFLoader("./data/BIC-4.0_2023_download.pdf")

documents = loader.load()

# Split de documenten in chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(documents)

from langchain_community.embeddings import OllamaEmbeddings

oembed = OllamaEmbeddings(base_url="http://pcloud:11434", model="nomic-embed-text")

# Maak een FAISS index aan
db = FAISS.from_documents(documents=all_splits, embedding=oembed)

# db = FAISS.from_documents(docs, OllamaEmbeddings(model="llama3"))
db.save_local("contexts/informatie_beveiliging_corporaties")
