import os
import torch
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.llms import HuggingFacePipeline
# ==============================
# LOAD ENV
# ==============================
load_dotenv()
access_token = os.getenv("HUGGINGFACE_HUB_ACCESS_TOKEN")

# ==============================
# LOAD EMBEDDINGS
# ==============================
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"}
)

# ==============================
# LOAD FAISS INDEX
# ==============================
vectorstore = FAISS.load_local(
    "vector_stores/faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# ==============================
# LOAD HUGGING FACE MODEL (Local)
# ==============================
model_name = "meta-llama/Llama-3"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype="auto"
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    return_full_text=False
)

llm = HuggingFacePipeline(pipeline=pipe)
# # ==============================
# # LOAD HUGGING FACE LLM
# # ==============================
# llm = HuggingFaceEndpoint(
#     repo_id="openai-community/gpt2",
#     task="text-generation",
#     huggingfacehub_api_token=access_token,
#     temperature=0.,
#     max_new_tokens=300
# )

chat = ChatHuggingFace(llm=llm)

# ==============================
# PROMPT TEMPLATE
# ==============================
template = """
You are an AI professor.

Answer ONLY using the context below.
If the answer is not present in the context, say:
"I don't know based on the provided material."

Do NOT generate additional questions.
Be concise and clear.

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

# ==============================
# CHAT LOOP
# ==============================
print("📘 ML Book RAG Ready (HuggingFace Endpoint)\n")

while True:
    query = input("Ask question (type exit to stop): ")

    if query.lower() == "exit":
        break

    response = rag_chain.invoke(query)
    print("\nAnswer:\n", response)
    print("\n" + "-"*50 + "\n")