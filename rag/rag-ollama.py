import os
import glob
from dotenv import load_dotenv
import gradio as gr

from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.chat_models import ChatOllama
from langchain_chroma import Chroma

import numpy as np
from sklearn.manifold import TSNE
import plotly.graph_objects as go
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceEmbeddings

MODEL = "llama3.2:1b"
db_name = "vector_db_ollama"

folders = glob.glob("./data/*")

documents = []
for folder in folders:
    doc_type = os.path.basename(folder)
    loader = DirectoryLoader(
        folder, glob="**/*.md", 
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True
    )
    folder_docs = loader.load()
    for doc in folder_docs:
        print(f"fDoc:{doc.metadata}")
        doc.metadata['doc_type'] = doc_type
        documents.append(doc)

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)

doc_types = set([doc.metadata['doc_type'] for doc in chunks])
print(f"Found {len(doc_types)} document types: {', '.join(doc_types)}")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

if os.path.exists(db_name):
    Chroma(persist_directory=db_name, embedding_function=embeddings).delete_collection()

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=db_name
)

collection = vectorstore._collection
print(f"Collection '{collection.name}' created with {collection.count()} documents.")

sample_embedding = collection.get(limit=1, include=["embeddings"])["embeddings"][0]
dimensions = len(sample_embedding)
print(f"Sample embedding dimensions: {dimensions}")

llm = ChatOllama(model_name=MODEL, temperature=0.7)
memory = ConversationBufferMemory(
    memory_key="chat_history", 
    return_messages=True)
retriever = vectorstore.as_retriever()
chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory
)

query = "Can you describe Insurellm in a few sentences?"
result = chain.invoke({"question": query})
print(f"Query: {query}")
print(f"Answer: {result['answer']}")

def chat(message, history):
    response = chain.invoke({"question": message})
    return response["answer"]

view = gr.ChatInterface(chat).launch()