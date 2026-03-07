import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
# ==============================
# CONFIG
# ==============================
VECTOR_STORE_PATH = "vector_stores/faiss_index"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# ==============================
# LOAD EMBEDDINGS
# ==============================
print("🔄 Loading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"}
)

# ==============================
# LOAD VECTOR STORE
# ==============================
print("🔄 Loading FAISS index...")

vectorstore = FAISS.load_local(
    VECTOR_STORE_PATH,
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# ==============================
# LOAD LLM
# ==============================
print("🔄 Loading LLM model... (This may take time)")

tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL,
    torch_dtype="auto",
    device_map="auto"
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    return_full_text=False
)

print("✅ AI-Prof RAG Ready!\n")

# ==============================
# CHAT LOOP
# ==============================
while True:
    query = input("Ask question (type exit to stop): ")

    if query.lower() == "exit":
        break

    # 1️⃣ Retrieve relevant documents
    docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])

    # 2️⃣ Create RAG prompt
    prompt = f"""
You are an AI professor.

Answer the question in 3-4 clear sentences.

Do NOT:
- Repeat the question
- Generate additional questions
- Continue the document
- Include the context in the answer

Context:
{context}

Question:
{query}

Answer:
"""

    # 3️⃣ Generate response
    response = pipe(
        prompt,
        max_new_tokens=150,
        temperature=0.1,
        do_sample=False
    )
    answer = response[0]["generated_text"]
    print("\n🔍 LLM Output:\n", answer)
# Remove prompt portion
    # answer = output[len(prompt):]
    
    print("\n📘 Answer:\n", answer.strip())
    # print("\n📘 Answer:\n")
    # print(response[0]["generated_text"])
    # print("\n" + "-" * 60 + "\n")