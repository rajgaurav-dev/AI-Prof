import os
import torch
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


# ==============================
# LOAD ENV VARIABLES
# ==============================
load_dotenv()

access_token = os.getenv("HUGGINGFACE_HUB_LATEST_ACCESS_TOKEN")

if access_token is None:
    raise ValueError("❌ HuggingFace token not found in .env file")

print("✅ Hugging Face Token Loaded")


# ==============================
# LOAD EMBEDDINGS
# ==============================
print("🔄 Loading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"}
)

print("✅ Embeddings Loaded")


# ==============================
# LOAD FAISS VECTOR STORE
# ==============================
print("🔄 Loading FAISS index...")

vectorstore = FAISS.load_local(
    "vector_stores/faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

print("✅ FAISS Loaded")


# ==============================
# LOAD HUGGING FACE MODEL (Cloud)
# ==============================
print("🔄 Connecting to HuggingFace Endpoint...")

llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    task="text-generation",
    huggingfacehub_api_token=access_token,
    temperature=0.2,
    max_new_tokens=1024
)

chat = ChatHuggingFace(llm=llm)

print("✅ LLM Connected")


# ==============================
# PROMPT TEMPLATE
# ==============================
template = """
You are an AI professor.

Answer ONLY using the context below.
If the answer is not present, say:
"I don't know based on the provided material."

Do NOT:
- Repeat the question
- Generate additional questions
- Continue the document
- Include the context in the answer

Give a clear explanation in 3–4 sentences using the context.
Do not repeat equations unless necessary.

Context:
{context}

Question:
{question}

Answer:
"""

prompt = ChatPromptTemplate.from_template(template)


# ==============================
# RAG CHAIN
# ==============================
rag_chain = (
    {
        "context": retriever,
        "question": RunnablePassthrough()
    }
    | prompt
    | chat
    | StrOutputParser()
)

print("\n📘 ML Book RAG Ready\n")


# ==============================
# CHAT LOOP WITH DEBUG
# ==============================
while True:
    query = input("Ask question (type exit to stop): ")

    if query.lower() == "exit":
        break

    # 🔎 DEBUG: Retrieve documents manually
    docs = retriever.invoke(query)

    print("\n🔎 Retrieved Chunks:\n")

    for i, doc in enumerate(docs):
        print(f"\n===== Chunk {i+1} =====")
        print(doc.page_content[:800])  # print first 800 characters
        print("=" * 50)

    # 🔎 Build context manually (for inspection)
    context = "\n\n".join(doc.page_content for doc in docs)

    print("\n📚 Final Context Sent To LLM:\n")
    print(context[:1500])  # limit to avoid flooding terminal
    print("\n" + "="*70 + "\n")

    # 🔥 Run RAG chain
    response = rag_chain.invoke(query)

    print("\n📘 Answer:\n", response)
    print("\n" + "-"*70 + "\n")