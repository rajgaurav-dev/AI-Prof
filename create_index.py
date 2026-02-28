import os
import torch
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

documents = []

for file in os.listdir("ML_Books"):
    if file.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join("ML_Books", file))
        documents.extend(loader.load())

# Split
splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,
    chunk_overlap=100,
    separators=["\n\n", "\n", ".", " "]
)

chunks = splitter.split_documents(documents)

# Embeddings
embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"}
    )
# Create FAISS index
vectorstore = FAISS.from_documents(chunks, embeddings)

# 🔥 SAVE TO DISK
vectorstore.save_local("vector_stores/faiss_index")

print("✅ Index created and saved!")