import os
import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
load_dotenv()

# 🔥 LOAD EXISTING INDEX
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"}
)

vectorstore = FAISS.load_local(
    "vector_stores/faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever()

llm = ChatOllama(model="llama3")

template = """
Answer the question based only on the following context:

{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

rag_chain = (
    {
        "context": retriever,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

while True:
    query = input("Ask question (exit to stop): ")

    if query.lower() == "exit":
        break

    response = rag_chain.invoke(query)
    print("\nAnswer:\n", response)