import os
import glob
from dotenv import load_dotenv
import gradio as gr

from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma

from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

MODEL = "gpt-4o-mini"
db_name = "vector_db"

load_dotenv(override=True)
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY', 'your-key-if-not-using-env')

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

embeddings = OpenAIEmbeddings()

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

# 1. Create the LLM
llm = ChatOpenAI(model_name=MODEL, temperature=0.7)
# 2. Create the memory
memory = ConversationBufferMemory(
    memory_key="chat_history", 
    return_messages=True)
# 3. Create the retriever
retriever = vectorstore.as_retriever()
# 4. Create the chain
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